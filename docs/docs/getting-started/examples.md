---
icon: octicons/code-16
---

# Examples

Let's create a simple "Hello World" simulation with a falling box.

## Step 1: Create a Python Script

Create a new file called `hello_world.py`:

```py
import mujoco
import mjswan

# Create a builder instance
builder = mjswan.Builder()

# Add a project
project = builder.add_project(name="Hello World")

# Define a simple MuJoCo model
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")

# Add a scene to the project
project.add_scene(model=model, name="Falling Box")

# Build the application
app = builder.build()

# Launch in browser
app.launch()
```

## Step 2: Run the Script

Execute your script:

```bash
python hello_world.py
```

This will:

1. Build the application in a `dist` directory
2. Start a local web server
3. Open your default browser to view the simulation

You should see an interactive 3D view with a green box falling onto a red plane!

## Troubleshooting

### Port Already in Use

If port 8080 is already in use, mjswan will automatically find an available port. You can also specify a custom port:

```python
app.launch(port=8888)
```

### Browser Doesn't Open

If the browser doesn't open automatically, you can manually navigate to the URL shown in the console output (typically `http://localhost:8080`).

### Build Directory

By default, the application is built to a `dist` directory relative to your script. You can specify a custom output directory:

```python
app = builder.build(output_dir="./my_app")
```

### WebAssembly Not Supported

mjswan requires a modern browser with WebAssembly support. Ensure you're using an up-to-date version of Chrome, Firefox, Safari, or Edge.
