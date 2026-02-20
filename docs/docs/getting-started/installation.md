---
icon: octicons/download-16
---

# Installation

mjswan can be installed using either Python (pip) or JavaScript (npm), depending on your preferred workflow.

<div class="grid cards" markdown>

-   [:simple-python: &nbsp; __Python package__](#python-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via pip to quickly build and share interactive MuJoCo simulations

-   [:simple-javascript: &nbsp; __JavaScript package__](#javascript-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via npm for custom web applications with TypeScript support

-   [:simple-github: &nbsp; __GitHub Source__](#github-source){ style="text-decoration: none; color: inherit;" }

    ---

    Clone the repository for development and contributing to the project

>   :simple-docker: &nbsp; __Docker / Cluster__
>   ---
>   Not supported.

</div>

## Requirements

- **Python**: Version 3.10 or higher
- **Node.js**: Version 20 or higher (for npm installation)
- **Browser**: Modern browser with WebAssembly and WebGL support

## Python Installation

Install mjswan with pip:

```bash
pip install mjswan
```

For development work, you can install with optional dependencies:

```bash
pip install 'mjswan[dev]'
```

For running examples:

```bash
pip install 'mjswan[examples]'
```

## JavaScript Installation

Install mjswan with npm:

```bash
npm install mjswan
```

Or with yarn:

```bash
yarn add mjswan
```

## GitHub Source

Clone the repository:

```bash
git clone https://github.com/ttktjmt/mjswan.git
cd mjswan
```

Install dependencies:

```bash
uv sync --all-extras
```
