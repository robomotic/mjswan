# SO101 Motor Physics and Control Modes in MuJoCo

To realistically model the STS3215 servos used in the SO101 dual-arm system, it is necessary to represent their electrical properties using **mechanical equivalents** in MuJoCo. Simple joint definitions and instantaneous position actuators result in a "floaty" or unrealistic feel when interacting with the simulation.

This document outlines the required parameters across both the literal `<joint>` and `<actuator>` elements, as well as how to configure the `mjswan` control policies.

## 1. Modeling Gearbox Resistance (Joint Properties)

A "Torque Off" servo is not completely free to move. Back-driving the high-reduction gearbox generates physical resistance. This is split into two components:

- **Stiction (`frictionloss`)**: The break-away torque required to overcome static gear friction. This is what prevents the arm from falling under its own weight when unpowered. For the STS3215, this ranges between **0.1 and 0.5 Nm**.
- **Viscous Friction (`damping`)**: Resistance proportional to speed, simulating grease and gear mesh drag. Values between **0.05 and 0.1 Nms/rad** are typical.

## 2. Rotor Inertia (Armature)

Even though the servo's rotor is tiny, its effective inertia is multiplied by the square of the gear ratio. This means the joint feels "heavy" to accelerate, independent of the link mass.

- **`armature`**: Add this to the `<joint>` configuration. For STS3215 servos acting through a ~1:300 gear ratio, values between **0.005 and 0.02** add the necessary "virtual mass."

## 3. Simulating Electrical Limits

The STS3215 has a stall torque of roughly 3.0 Nm (30kg·cm) at 7.4V. The model should prevent infinite torque outputs when obstacles are hit or targets are out of reach.

- **`forcerange`**: Defined on the `<actuator>`. For our variants, this limits the pure torque output based on hardware limits.

## 4. mjswan Control Modes (Gravity and Force Hook)

To implement a "Gravity / Torqueless" mode where the arm sags naturally but holds its position due to stiction:

- Do **not** use `JointPositionActionCfg(stiffness=0)`. Position configurations are interpreted as springs, and even with 0 stiffness, unexpected offsets or velocity drag artifacts can pull the arm towards `q_default`.
- **Use `JointEffortActionCfg`:** This configures the runtime to pass torque directly into the MuJoCo `ctrl` array. A `ctrl=0` output from the ONNX policy precisely means zero commanded torque, allowing gravity, `frictionloss`, and `damping` to handle the rest.
