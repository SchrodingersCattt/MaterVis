# MaterVis - Crystal Structure Visualizer

A lightweight, in-browser crystal structure visualizer with a Materials Project-like aesthetic and hard plastic gloss rendering.

## Features

- **Input Formats**: CIF, POSCAR/CONTCAR (VASP), and JSON
- **Rendering**: Atoms as glossy plastic spheres, bonds as cylinders, coordination polyhedra
- **Materials**: Hard plastic appearance using PBR with Three.js MeshPhysicalMaterial
- **Interactivity**: Orbital camera, atom selection, disorder visualization
- **Analysis**: Bonding calculations, polyhedra identification
- **Exports**: PNG screenshots, GLB models, session JSON

## Tech Stack

- **TypeScript** + **Vite**
- **WebGL via Three.js**
- **State Management**: Zustand
- **Unit Tests**: Vitest
- **Linting/Formatting**: ESLint + Prettier

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Project Structure

```
src/
├── parsers/           # File format parsers (CIF, POSCAR, JSON)
├── rendering/         # Rendering components (atoms, bonds, polyhedra)
├── utils/             # Utility functions (lattice math, bonding)
├── crystal.ts         # Crystal data models
├── store.ts           # State management
├── viewer.ts          # Main viewer component
├── renderer.ts        # Crystal structure renderer
└── main.ts            # Application entry point
```

## Implementation Status

This is a work in progress. Currently implemented:

- Basic Three.js viewer with plastic material rendering
- File parsers for CIF, POSCAR, and JSON formats
- Atom, bond, and polyhedra rendering systems
- State management with Zustand
- Core crystal structure data models

## Planned Features

- Full periodic boundary conditions support
- Coordination polyhedra with convex hull computation
- Disorder visualization with occupancy sliders
- Crystallographic tools (supercell builder, symmetry operations)
- UI panels for controlling visualization parameters
- Export functionality (PNG, GLB, JSON)
- Performance optimizations for large structures

## License

MIT