"""Robot-agnostic grasp description and its RH56F1 / xArm7 consumers, without weights or MANO files."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sample import grasp_interface as gi
from sample import rh56f1_retarget as rh
from sample.validate_rh56f1 import PalmPrior, opposing_order, wrist_table_clearance
from sample.validate_xarm7_rh56f1 import APPROACH_LENGTH, pregrasp_candidates


def hand_joints(rotation=np.eye(3), origin=np.zeros(3), aperture=.08):
    """A schematic right hand: wrist at the origin, fingers along +z, index toward +y."""
    joints = np.zeros((21, 3))
    joints[gi.WRIST] = [0, 0, 0]
    joints[gi.INDEX_MCP] = [0, .03, .09]
    joints[gi.MIDDLE_MCP] = [0, .01, .095]
    joints[gi.RING_MCP] = [0, -.01, .09]
    joints[gi.PINKY_MCP] = [0, -.03, .085]
    joints[gi.INDEX_TIP] = [.02, .03, .17]
    joints[gi.MIDDLE_TIP] = [.02, .01, .18]
    joints[gi.THUMB_TIP] = joints[[gi.INDEX_TIP, gi.MIDDLE_TIP]].mean(axis=0) + [aperture, 0, 0]
    return joints @ rotation.T + origin


def approaching_hand(frames=50, travel=(0, -.2, 0), apertures=None):
    """Hand translating by ``travel`` with a per-frame grip aperture."""
    apertures = np.full(frames, .03) if apertures is None else np.asarray(apertures)
    path = np.linspace(-np.asarray(travel, dtype=float), 0, frames)
    return np.stack([hand_joints(origin=p, aperture=a) for p, a in zip(path, apertures)])


class PalmFrameTest(unittest.TestCase):
    def test_axes_follow_the_hand(self):
        rotation = Rotation.from_euler('xyz', [20, -35, 70], degrees=True).as_matrix()
        origin = np.array([.1, -.2, .3])
        palm = gi.palm_frame(hand_joints(rotation, origin))
        reference = gi.palm_frame(hand_joints())
        np.testing.assert_allclose(palm[:3, :3], rotation @ reference[:3, :3], atol=1e-9)
        np.testing.assert_allclose(palm[:3, 3], rotation @ reference[:3, 3] + origin, atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(palm[:3, :3]), 1.)
        # +z points to the fingers and +y toward the index side.
        self.assertGreater(reference[2, 2], .99)
        self.assertGreater(reference[1, 1], .99)

    def test_deviation_reports_offset_in_reference_axes(self):
        reference = gi.palm_frame(hand_joints(Rotation.from_euler('z', 90, degrees=True).as_matrix()))
        moved = reference.copy()
        moved[:3, 3] += reference[:3, :3] @ [.01, .02, -.03]
        moved[:3, :3] = moved[:3, :3] @ Rotation.from_euler('y', 25, degrees=True).as_matrix()
        offset, angle = gi.palm_deviation(reference, moved)
        np.testing.assert_allclose(offset, [.01, .02, -.03], atol=1e-12)
        self.assertAlmostEqual(angle, 25., places=6)

    def test_robot_palm_matches_definition(self):
        palm = rh.palm_in_flange(rh.HandModel())
        np.testing.assert_allclose(palm[:3, 2], [0, 0, 1], atol=1e-9)
        self.assertGreater(palm[1, 1], .99)  # little -> index is +y on the RH56F1
        self.assertTrue(.15 < palm[2, 3] < .17)


class ApproachAndPreshapeTest(unittest.TestCase):
    def test_approach_direction_is_palm_travel(self):
        spec = gi.extract_spec(approaching_hand(travel=(0, -.2, 0)))
        self.assertTrue(spec['approach_valid'])
        np.testing.assert_allclose(spec['approach_dir_object'], [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(spec['approach_dir_palm'], spec['T_object_palm'][:3, :3].T @ [0, -1, 0])

    def test_stationary_hand_has_no_approach(self):
        spec = gi.extract_spec(approaching_hand(travel=(0, 0, 0)))
        self.assertFalse(spec['approach_valid'])
        np.testing.assert_array_equal(spec['approach_dir_object'], np.zeros(3))

    def test_preshape_against_remaining_distance(self):
        frames = 50
        # Opens from 3 cm to 9 cm, then closes to 4 cm at the grasp.
        apertures = np.interp(np.arange(frames), [0, 30, 49], [.03, .09, .04])
        spec = gi.extract_spec(approaching_hand(frames, (0, -.2, 0), apertures))
        self.assertAlmostEqual(spec['remaining_m'][0], .2, places=6)
        self.assertEqual(spec['remaining_m'][-1], 0.)
        self.assertEqual(spec['preshape'][-1], 0.)
        self.assertAlmostEqual(spec['preshape'].max(), 1.)
        widest = spec['remaining_m'][30]
        self.assertAlmostEqual(gi.preshape_at(widest, spec['remaining_m'], spec['preshape']), 1., places=6)
        # Beyond the recorded path the first frame's value holds.
        self.assertAlmostEqual(gi.preshape_at(1., spec['remaining_m'], spec['preshape']), spec['preshape'][0])

    def test_small_aperture_change_gives_flat_preshape(self):
        apertures = np.linspace(.040, .043, 50)
        spec = gi.extract_spec(approaching_hand(apertures=apertures))
        np.testing.assert_array_equal(spec['preshape'], np.zeros(50))

    def test_contacts_from_finger_distances(self):
        spec = gi.extract_spec(approaching_hand(), finger_distance=[.0005, .002, .0008, .01, .0009],
                               contact_radius=.001)
        self.assertEqual(tuple(spec['contact_fingers']), ('thumb', 'middle', 'pinky'))
        self.assertEqual(opposing_order(spec['contact_fingers']), ('middle', 'index', 'ring'))

    def test_preshape_joints_keep_thumb_rotation(self):
        q_open = np.array([.8, .2, .3, .4, .5, .6])
        np.testing.assert_allclose(rh.preshape_joints(q_open, 0.), q_open)
        wide = rh.preshape_joints(q_open, 1.)
        self.assertEqual(wide[0], .8)
        np.testing.assert_array_equal(wide[1:], 0.)
        np.testing.assert_allclose(rh.preshape_joints(q_open, [.5])[0, 1:], q_open[1:] / 2)


class PalmPriorTest(unittest.TestCase):
    def setUp(self):
        self.flange_palm = rh.palm_in_flange(rh.HandModel())
        self.human = gi.palm_frame(hand_joints(Rotation.from_euler('x', 30, degrees=True).as_matrix(), [.0, -.1, .07]))

    def test_aligned_flange_reproduces_human_palm(self):
        prior = PalmPrior(self.human, self.flange_palm)
        flange = prior.initial_flange()
        offset, angle = prior.deviation(flange)
        np.testing.assert_allclose(offset, 0, atol=1e-12)
        self.assertAlmostEqual(angle, 0., places=6)
        np.testing.assert_allclose(prior.residual(flange), 0, atol=1e-12)
        self.assertTrue(prior.within(flange))

    def test_flexion_has_its_own_budget(self):
        prior = PalmPrior(self.human, self.flange_palm, max_rotation_deg=30, max_flexion_deg=60, max_offset_m=1)
        aligned = prior.initial_flange()

        def rotated(axis, degrees):
            # Rotate the robot palm about a human palm axis through the palm origin.
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(np.deg2rad(degrees) * self.human[:3, axis]).as_matrix()
            pivot = np.eye(4)
            pivot[:3, 3] = self.human[:3, 3]
            return pivot @ turn @ np.linalg.inv(pivot) @ aligned

        flexion, other = prior.rotation_parts(rotated(1, 50))
        self.assertAlmostEqual(flexion, 50., places=5)
        self.assertAlmostEqual(other, 0., places=5)
        self.assertTrue(prior.within(rotated(1, 50)))
        self.assertFalse(prior.within(rotated(1, 70)))
        self.assertFalse(prior.within(rotated(2, 40)))

    def test_wrist_clearance_depends_on_finger_elevation(self):
        flange = np.eye(4)
        flange[2, 3] = .10
        self.assertAlmostEqual(wrist_table_clearance(flange, 0.), -.06)  # fingers up: segment hangs below
        down = flange.copy()
        down[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        self.assertAlmostEqual(wrist_table_clearance(down, 0.), .04)  # fingers down: flange is lowest
        level = flange.copy()
        level[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
        self.assertAlmostEqual(wrist_table_clearance(level, 0.), .04)


class PregraspTest(unittest.TestCase):
    def test_flowhoi_direction_first_then_flange_axis(self):
        goal = np.eye(4)
        goal[:3, 3] = [.5, 0, .1]
        candidates = pregrasp_candidates(goal, np.array([0, 0, 1.]), approach_dir=np.array([1., 0, 0]))
        first, source, deviation = candidates[0]
        self.assertEqual(source, 'flowhoi')
        self.assertEqual(deviation, 0.)
        np.testing.assert_allclose(first[:3, 3], goal[:3, 3] - [APPROACH_LENGTH, 0, 0], atol=1e-12)
        self.assertEqual([c[1] for c in candidates].count('flowhoi'), 7)
        self.assertEqual(candidates[7][1], 'flange_axis')
        np.testing.assert_allclose(candidates[7][0][:3, 3], goal[:3, 3] - [0, 0, APPROACH_LENGTH], atol=1e-12)

    def test_without_flowhoi_only_flange_axis(self):
        candidates = pregrasp_candidates(np.eye(4), np.array([0, 0, 1.]))
        self.assertEqual({c[1] for c in candidates}, {'flange_axis'})
        self.assertIsNone(candidates[0][2])


if __name__ == '__main__':
    unittest.main()
