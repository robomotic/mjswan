---
icon: octicons/light-bulb-16
---

# Core Concepts

This guide introduces the core concepts and architecture of mjswan.

## Architecture Overview

mjswan follows a hierarchical structure:

```
Builder
  └── Project(s)
        └── Scene(s)
              └── Policy (optional)
```

### Builder

The `Builder` is the top-level object that orchestrates the entire application. It manages projects and handles the build process.

```python
builder = mjswan.Builder(base_path="/")
```

**Key responsibilities:**

- Managing multiple projects
- Configuring deployment paths
- Building the final application

### Project

A `Project` is a collection of related scenes. Projects help organize your simulations logically and appear as sections in the web interface.

```python
project = builder.add_project(name="Robot Experiments", id="robots")
```

**Key features:**

- Groups related scenes together
- Has a display name shown in the UI
- Has an optional ID for URL routing
- First project is the default/home page

### Scene

A `Scene` represents a single simulation instance with a MuJoCo model and optional policy.

```python
scene = project.add_scene(
    model=mujoco_model,
    name="Walking Gait",
    initial_qpos={...}
)
```

**Key features:**

- Contains a MuJoCo physics model
- Can have custom initial states
- May include an ONNX policy for control
- Appears as a selectable option in the UI

### Policy

A `Policy` is an ONNX neural network that controls the simulation based on observations.

```python
scene.add_policy(
    path="policy.onnx",
    observation_config={...},
    action_config={...}
)
```

**Key features:**

- Runs inference in real-time in the browser
- Processes observations from the simulation
- Outputs actions to control the model
- Fully client-side (no server required)
