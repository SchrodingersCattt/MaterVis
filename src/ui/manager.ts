/**
 * UI Manager for the crystal visualizer
 */

import { useStore } from '../store'
import { Viewer } from '../viewer'
import { UIPanel, ControlPanel, createCheckbox, createSlider, createButton } from './panels'

export class UIManager {
  private container: HTMLDivElement
  private viewer: Viewer
  
  constructor(viewer: Viewer) {
    this.viewer = viewer
    
    // Create UI container
    this.container = document.createElement('div')
    this.container.className = 'ui-container'
    document.getElementById('app')?.appendChild(this.container)
    
    // Create panels
    this.createStructurePanel()
    this.createAppearancePanel()
    this.createBondingPanel()
    this.createExportPanel()
    
    // Update UI when state changes
    useStore.subscribe(this.updateUI.bind(this))
  }
  
  private createStructurePanel(): void {
    const panel = new ControlPanel('Structure')
    
    // Supercell controls
    const supercellX = createSlider('Supercell X', 1, 5, 1, 1, (value) => {
      const [x, y, z] = useStore.getState().supercell
      useStore.getState().setSupercell([value, y, z])
    })
    
    const supercellY = createSlider('Supercell Y', 1, 5, 1, 1, (value) => {
      const [x, y, z] = useStore.getState().supercell
      useStore.getState().setSupercell([x, value, z])
    })
    
    const supercellZ = createSlider('Supercell Z', 1, 5, 1, 1, (value) => {
      const [x, y, z] = useStore.getState().supercell
      useStore.getState().setSupercell([x, y, value])
    })
    
    panel.addControl(supercellX)
    panel.addControl(supercellY)
    panel.addControl(supercellZ)
    
    // Axes toggle
    const axesToggle = createCheckbox('Show Axes', useStore.getState().showAxes, (checked) => {
      useStore.getState().setShowAxes(checked)
    })
    
    panel.addControl(axesToggle)
    
    this.container.appendChild(panel.getElement())
  }
  
  private createAppearancePanel(): void {
    const panel = new ControlPanel('Appearance')
    
    // Bonds toggle
    const bondsToggle = createCheckbox('Show Bonds', useStore.getState().showBonds, (checked) => {
      useStore.getState().setShowBonds(checked)
    })
    
    // Polyhedra toggle
    const polyhedraToggle = createCheckbox('Show Polyhedra', useStore.getState().showPolyhedra, (checked) => {
      useStore.getState().setShowPolyhedra(checked)
    })
    
    panel.addControl(bondsToggle)
    panel.addControl(polyhedraToggle)
    
    this.container.appendChild(panel.getElement())
  }
  
  private createBondingPanel(): void {
    const panel = new ControlPanel('Bonding')
    
    // Bonding scale factor
    const scaleSlider = createSlider(
      'Scale Factor', 
      0.8, 
      1.5, 
      useStore.getState().bonding.f, 
      0.01,
      (value) => {
        const bonding = useStore.getState().bonding
        useStore.getState().setBonding({ ...bonding, f: value })
      }
    )
    
    panel.addControl(scaleSlider)
    
    this.container.appendChild(panel.getElement())
  }
  
  private createExportPanel(): void {
    const panel = new ControlPanel('Export')
    
    // Screenshot button
    const screenshotBtn = createButton('Take Screenshot', () => {
      // In a full implementation, this would trigger a screenshot
      console.log('Screenshot requested')
    })
    
    // Export GLB button
    const exportGLBBtn = createButton('Export GLB', () => {
      // In a full implementation, this would export the scene as GLB
      console.log('GLB export requested')
    })
    
    panel.addControl(screenshotBtn)
    panel.addControl(exportGLBBtn)
    
    this.container.appendChild(panel.getElement())
  }
  
  private updateUI(): void {
    // In a full implementation, this would update UI elements based on state
    // For now, we'll just log state changes
    console.log('State updated:', useStore.getState())
  }
}