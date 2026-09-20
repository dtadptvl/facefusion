import Foundation
import simd

/// Geometry, rotational kinematics, and expression manipulation for LivePortrait processors (face_editor and expression_restorer).
public enum LivePortraitGeometry {

    public static let expressionMin: [[Float]] = [
        [-2.88067125e-02, -8.12731311e-02, -1.70541159e-03],
        [-4.88598682e-02, -3.32196616e-02, -1.67431499e-04],
        [-6.75425082e-02, -4.28681746e-02, -1.98950816e-04],
        [-7.23103955e-02, -3.28503326e-02, -7.31324719e-04],
        [-3.87073644e-02, -6.01546466e-02, -5.50269964e-04],
        [-6.38048723e-02, -2.23840728e-01, -7.13261834e-04],
        [-3.02710701e-02, -3.93195450e-02, -8.24086510e-06],
        [-2.95799859e-02, -5.39318882e-02, -1.74219604e-04],
        [-2.92359516e-02, -1.53050944e-02, -6.30460854e-05],
        [-5.56493877e-03, -2.34344602e-02, -1.26858242e-04],
        [-4.37593013e-02, -2.77768299e-02, -2.70503685e-02],
        [-1.76926646e-02, -1.91676542e-02, -1.15090821e-04],
        [-8.34268332e-03, -3.99775570e-03, -3.27481248e-05],
        [-3.40162888e-02, -2.81868968e-02, -1.96679524e-04],
        [-2.91855410e-02, -3.97511162e-02, -2.81230678e-05],
        [-1.50395725e-02, -2.49494594e-02, -9.42573533e-05],
        [-1.67938769e-02, -2.00953931e-02, -4.00750607e-04],
        [-1.86435618e-02, -2.48535164e-02, -2.74416432e-02],
        [-4.61211195e-03, -1.21660791e-02, -2.93173041e-04],
        [-4.10017073e-02, -7.43824020e-02, -4.42762971e-02],
        [-1.90370996e-02, -3.74363363e-02, -1.34740388e-02]
    ]

    public static let expressionMax: [[Float]] = [
        [4.46682945e-02, 7.08772913e-02, 4.08344204e-04],
        [2.14308221e-02, 6.15894832e-02, 4.85319615e-05],
        [3.02363783e-02, 4.45043296e-02, 1.28298725e-05],
        [3.05869691e-02, 3.79812494e-02, 6.57040102e-04],
        [4.45670523e-02, 3.97259220e-02, 7.10966764e-04],
        [9.43699256e-02, 9.85926315e-02, 2.02551950e-04],
        [1.61131397e-02, 2.92906128e-02, 3.44733417e-06],
        [5.23825921e-02, 1.07065082e-01, 6.61510974e-04],
        [2.85718683e-03, 8.32320191e-03, 2.39314613e-04],
        [2.57947259e-02, 1.60935968e-02, 2.41853559e-05],
        [4.90833223e-02, 3.43903080e-02, 3.22353356e-02],
        [1.44766076e-02, 3.39248963e-02, 1.42291479e-04],
        [8.75749043e-04, 6.82212645e-03, 2.76097053e-05],
        [1.86958015e-02, 3.84016186e-02, 7.33085908e-05],
        [2.01714113e-02, 4.90544215e-02, 2.34028921e-05],
        [2.46518422e-02, 3.29151377e-02, 3.48571630e-05],
        [2.22457591e-02, 1.21796541e-02, 1.56396593e-04],
        [1.72109623e-02, 3.01626958e-02, 1.36556877e-02],
        [1.83460284e-02, 1.61141958e-02, 2.87440169e-04],
        [3.57594155e-02, 1.80554688e-01, 2.75554154e-02],
        [2.17450950e-02, 8.66811201e-02, 3.34241726e-02]
    ]

    /// Linear interpolation helper matching numpy.interp.
    public static func interp(x: Float, xp0: Float, xp1: Float, fp0: Float, fp1: Float) -> Float {
        if x <= xp0 { return fp0 }
        if x >= xp1 { return fp1 }
        let t = (x - xp0) / (xp1 - xp0)
        return fp0 + t * (fp1 - fp0)
    }

    /// Creates 3x3 extrinsic Euler XYZ rotation matrix matching Scipy's Rotation.from_euler('xyz', ..., degrees=True).
    /// R = Rz(roll) * Ry(yaw) * Rx(pitch).
    public static func createRotation(pitch: Float, yaw: Float, roll: Float) -> simd_float3x3 {
        let p = pitch * Float.pi / 180.0
        let y = yaw * Float.pi / 180.0
        let r = roll * Float.pi / 180.0

        let cp = cos(p), sp = sin(p)
        let cy = cos(y), sy = sin(y)
        let cr = cos(r), sr = sin(r)

        let r00 = cr * cy
        let r01 = sp * sy * cr - sr * cp
        let r02 = sp * sr + sy * cp * cr

        let r10 = sr * cy
        let r11 = sp * sr * sy + cp * cr
        let r12 = -sp * cr + sr * sy * cp

        let r20 = -sy
        let r21 = sp * cy
        let r22 = cp * cy

        // simd_float3x3 is column-major: columns are (col0, col1, col2)
        let col0 = SIMD3<Float>(r00, r10, r20)
        let col1 = SIMD3<Float>(r01, r11, r21)
        let col2 = SIMD3<Float>(r02, r12, r22)
        return simd_float3x3(columns: (col0, col1, col2))
    }

    /// Calculates Euler angle limits for LivePortrait head pose editing.
    public static func calculateEulerLimits(pitch: Float, yaw: Float, roll: Float) -> (pitchMin: Float, pitchMax: Float, yawMin: Float, yawMax: Float, rollMin: Float, rollMax: Float) {
        var pitchMin: Float = -30.0
        var pitchMax: Float = 30.0
        var yawMin: Float = -60.0
        var yawMax: Float = 60.0
        var rollMin: Float = -20.0
        var rollMax: Float = 20.0

        if pitch < 0 { pitchMin = min(pitch, pitchMin) } else { pitchMax = max(pitch, pitchMax) }
        if yaw < 0 { yawMin = min(yaw, yawMin) } else { yawMax = max(yaw, yawMax) }
        if roll < 0 { rollMin = min(roll, rollMin) } else { rollMax = max(roll, rollMax) }

        return (pitchMin, pitchMax, yawMin, yawMax, rollMin, rollMax)
    }

    /// Clamps output Euler angles within limits calculated from target angles.
    public static func limitAngle(targetPitch: Float, targetYaw: Float, targetRoll: Float, outputPitch: Float, outputYaw: Float, outputRoll: Float) -> (pitch: Float, yaw: Float, roll: Float) {
        let limits = calculateEulerLimits(pitch: targetPitch, yaw: targetYaw, roll: targetRoll)
        let p = min(max(outputPitch, limits.pitchMin), limits.pitchMax)
        let y = min(max(outputYaw, limits.yawMin), limits.yawMax)
        let r = min(max(outputRoll, limits.rollMin), limits.rollMax)
        return (p, y, r)
    }

    /// Clamps (21, 3) expression tensor within empirical min/max bounds.
    public static func limitExpression(_ expression: inout [[Float]]) {
        for i in 0..<min(21, expression.count) {
            for j in 0..<3 {
                expression[i][j] = min(max(expression[i][j], expressionMin[i][j]), expressionMax[i][j])
            }
        }
    }

    /// Computes edited head rotation matrix from base pose and delta sliders in [-1, 1].
    public static func editHeadRotation(pitch: Float, yaw: Float, roll: Float, editPitchSlider: Float, editYawSlider: Float, editRollSlider: Float) -> simd_float3x3 {
        let dp = interp(x: editPitchSlider, xp0: -1, xp1: 1, fp0: 20, fp1: -20)
        let dy = interp(x: editYawSlider, xp0: -1, xp1: 1, fp0: 60, fp1: -60)
        let dr = interp(x: editRollSlider, xp0: -1, xp1: 1, fp0: -15, fp1: 15)

        let targetP = pitch + dp
        let targetY = yaw + dy
        let targetR = roll + dr

        let limited = limitAngle(targetPitch: pitch, targetYaw: yaw, targetRoll: roll, outputPitch: targetP, outputYaw: targetY, outputRoll: targetR)
        return createRotation(pitch: limited.pitch, yaw: limited.yaw, roll: limited.roll)
    }

    /// Modifies expression parameters based on user slider controls.
    public static func applyExpressionSliders(
        expression: inout [[Float]],
        eyebrowDirection: Float,
        eyeGazeHorizontal: Float,
        eyeGazeVertical: Float,
        mouthGrim: Float,
        mouthPout: Float,
        mouthPurse: Float,
        mouthSmile: Float,
        mouthPosHorizontal: Float,
        mouthPosVertical: Float
    ) {
        // Eyebrow
        if eyebrowDirection > 0 {
            expression[1][1] += interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
            expression[2][1] -= interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.020, fp1: 0.020)
        } else if eyebrowDirection < 0 {
            expression[1][0] -= interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
            expression[2][0] += interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.020, fp1: 0.020)
            expression[1][1] += interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
            expression[2][1] -= interp(x: eyebrowDirection, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
        }

        // Eye Gaze
        if eyeGazeHorizontal > 0 {
            expression[11][0] += interp(x: eyeGazeHorizontal, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
            expression[15][0] += interp(x: eyeGazeHorizontal, xp0: -1, xp1: 1, fp0: -0.020, fp1: 0.020)
        } else if eyeGazeHorizontal < 0 {
            expression[11][0] += interp(x: eyeGazeHorizontal, xp0: -1, xp1: 1, fp0: -0.020, fp1: 0.020)
            expression[15][0] += interp(x: eyeGazeHorizontal, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
        }
        if eyeGazeVertical != 0 {
            expression[1][1] += interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.0025, fp1: 0.0025)
            expression[2][1] -= interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.0025, fp1: 0.0025)
            expression[11][1] -= interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.010, fp1: 0.010)
            expression[13][1] -= interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
            expression[15][1] -= interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.010, fp1: 0.010)
            expression[16][1] -= interp(x: eyeGazeVertical, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
        }

        // Mouth Grim
        if mouthGrim > 0 {
            expression[17][2] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
            expression[19][2] += interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.01, fp1: 0.01)
            expression[20][1] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.06, fp1: 0.06)
            expression[20][2] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.03, fp1: 0.03)
        } else if mouthGrim < 0 {
            expression[19][1] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.05, fp1: 0.05)
            expression[19][2] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
            expression[20][2] -= interp(x: mouthGrim, xp0: -1, xp1: 1, fp0: -0.03, fp1: 0.03)
        }

        // Mouth Position
        if mouthPosHorizontal != 0 {
            expression[19][0] += interp(x: mouthPosHorizontal, xp0: -1, xp1: 1, fp0: -0.05, fp1: 0.05)
            expression[20][0] += interp(x: mouthPosHorizontal, xp0: -1, xp1: 1, fp0: -0.04, fp1: 0.04)
        }
        if mouthPosVertical > 0 {
            expression[19][1] -= interp(x: mouthPosVertical, xp0: -1, xp1: 1, fp0: -0.04, fp1: 0.04)
            expression[20][1] -= interp(x: mouthPosVertical, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
        } else if mouthPosVertical < 0 {
            expression[19][1] -= interp(x: mouthPosVertical, xp0: -1, xp1: 1, fp0: -0.05, fp1: 0.05)
            expression[20][1] -= interp(x: mouthPosVertical, xp0: -1, xp1: 1, fp0: -0.04, fp1: 0.04)
        }

        // Mouth Pout
        if mouthPout > 0 {
            expression[19][1] -= interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.022, fp1: 0.022)
            expression[19][2] += interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.025, fp1: 0.025)
            expression[20][2] -= interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.002, fp1: 0.002)
        } else if mouthPout < 0 {
            expression[19][1] += interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.022, fp1: 0.022)
            expression[19][2] += interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.025, fp1: 0.025)
            expression[20][2] -= interp(x: mouthPout, xp0: -1, xp1: 1, fp0: -0.002, fp1: 0.002)
        }

        // Mouth Purse
        if mouthPurse > 0 {
            expression[19][1] -= interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.04, fp1: 0.04)
            expression[19][2] -= interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
        } else if mouthPurse < 0 {
            expression[14][1] -= interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
            expression[17][2] += interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.01, fp1: 0.01)
            expression[19][2] -= interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
            expression[20][2] -= interp(x: mouthPurse, xp0: -1, xp1: 1, fp0: -0.002, fp1: 0.002)
        }

        // Mouth Smile
        if mouthSmile > 0 {
            expression[20][1] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.015, fp1: 0.015)
            expression[14][1] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.025, fp1: 0.025)
            expression[17][1] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.01, fp1: 0.01)
            expression[17][2] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.004, fp1: 0.004)
            expression[3][1] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.0045, fp1: 0.0045)
            expression[7][1] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.0045, fp1: 0.0045)
        } else if mouthSmile < 0 {
            expression[14][1] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
            expression[17][1] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.003, fp1: 0.003)
            expression[19][1] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.02, fp1: 0.02)
            expression[19][2] -= interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.005, fp1: 0.005)
            expression[20][2] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.01, fp1: 0.01)
            expression[3][1] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.0045, fp1: 0.0045)
            expression[7][1] += interp(x: mouthSmile, xp0: -1, xp1: 1, fp0: -0.0045, fp1: 0.0045)
        }

        limitExpression(&expression)
    }

    /// Transforms (21, 3) canonical motion points using scale, rotation, expression, and translation:
    /// transformed = scale * (points * rotation^T + expression) + translation
    public static func transformMotionPoints(
        points: [[Float]],
        rotation: simd_float3x3,
        expression: [[Float]],
        scale: Float,
        translation: SIMD3<Float>
    ) -> [[Float]] {
        var transformed = [[Float]](repeating: [Float](repeating: 0, count: 3), count: points.count)
        for i in 0..<points.count {
            let pt = SIMD3<Float>(points[i][0], points[i][1], points[i][2])
            // pt * rotation^T == rotation * pt
            let rotated = rotation * pt
            let exp = SIMD3<Float>(expression[i][0], expression[i][1], expression[i][2])
            let res = (rotated + exp) * scale + translation
            transformed[i] = [res.x, res.y, res.z]
        }
        return transformed
    }
}
