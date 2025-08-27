import './style.css'
import { Viewer } from './viewer'
import { createSiCrystal } from './demo'

const app = document.querySelector<HTMLDivElement>('#app')!

// Create the viewer instance
const viewer = new Viewer(app)

// Load a demo structure
const siCrystal = createSiCrystal()
viewer.loadCrystal(siCrystal)

// For demo purposes, we'll just log that the app is running
console.log('MaterVis - Crystal Structure Visualizer')
console.log('Viewer initialized:', viewer)