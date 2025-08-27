/**
 * Main renderer class that manages all visualization components
 */

import * as THREE from 'three'
import { CrystalStructure, Site } from './crystal'
import { createInstancedAtoms } from './rendering/atoms'
import { calculateBonds } from './utils/bonds'
import { createInstancedBonds } from './rendering/bonds'
import { createPolyhedronMesh, identifyPolyhedronType } from './rendering/polyhedra'
import { useStore } from './store'

export class CrystalRenderer {
  private scene: THREE.Scene
  private crystal: CrystalStructure | null = null
  private atoms: THREE.InstancedMesh | null = null
  private bonds: THREE.InstancedMesh | null = null
  private polyhedra: THREE.Group[] = []
  
  constructor(scene: THREE.Scene) {
    this.scene = scene
  }
  
  /**
   * Load and render a crystal structure
   * @param crystal Crystal structure to render
   */
  public loadCrystal(crystal: CrystalStructure): void {
    this.crystal = crystal
    
    // Clear previous visualization
    this.clear()
    
    // Render atoms
    this.renderAtoms()
    
    // Render bonds
    this.renderBonds()
    
    // Render polyhedra (initially hidden)
    this.renderPolyhedra()
  }
  
  /**
   * Render atoms using instanced mesh for performance
   */
  private renderAtoms(): void {
    if (!this.crystal) return
    
    this.atoms = createInstancedAtoms(this.crystal.sites, 0.5)
    this.scene.add(this.atoms)
  }
  
  /**
   * Render bonds using instanced mesh for performance
   */
  private renderBonds(): void {
    if (!this.crystal) return
    
    const state = useStore.getState()
    const bonds = calculateBonds(
      this.crystal,
      state.bonding.f,
      state.bonding.min,
      state.bonding.max,
      state.bonding.pairOverrides
    )
    
    this.bonds = createInstancedBonds(bonds, this.crystal.sites, 0.05)
    
    if (state.showBonds) {
      this.scene.add(this.bonds)
    }
  }
  
  /**
   * Render coordination polyhedra
   */
  private renderPolyhedra(): void {
    if (!this.crystal) return
    
    const state = useStore.getState()
    
    // Clear existing polyhedra
    this.polyhedra.forEach(p => this.scene.remove(p))
    this.polyhedra = []
    
    if (!state.showPolyhedra) return
    
    // For demonstration, create polyhedra for the first few atoms
    const maxPolyhedra = Math.min(5, this.crystal.sites.length)
    
    for (let i = 0; i < maxPolyhedra; i++) {
      const center = this.crystal.sites[i]
      
      // Find neighbors (simplified - in a real implementation we would use the bonding data)
      const neighbors: Site[] = []
      const maxNeighbors = 6
      
      for (let j = 0; j < this.crystal.sites.length && neighbors.length < maxNeighbors; j++) {
        if (i !== j) {
          neighbors.push(this.crystal.sites[j])
        }
      }
      
      if (neighbors.length >= 3) {
        const polyhedron = createPolyhedronMesh(center, neighbors, 0.3)
        this.polyhedra.push(polyhedron)
        this.scene.add(polyhedron)
        
        // Log polyhedron type
        const type = identifyPolyhedronType(neighbors)
        console.log(`Atom ${i} (${center.element}): ${type} polyhedron`)
      }
    }
  }
  
  /**
   * Update visualization based on state changes
   */
  public update(): void {
    if (!this.crystal) return
    
    const state = useStore.getState()
    
    // Update bonds visibility
    if (this.bonds) {
      if (state.showBonds && !this.bonds.parent) {
        this.scene.add(this.bonds)
      } else if (!state.showBonds && this.bonds.parent) {
        this.scene.remove(this.bonds)
      }
    }
    
    // Update polyhedra visibility
    const showingPolyhedra = this.polyhedra.some(p => p.parent)
    if (state.showPolyhedra && !showingPolyhedra) {
      this.renderPolyhedra()
    } else if (!state.showPolyhedra && showingPolyhedra) {
      this.polyhedra.forEach(p => this.scene.remove(p))
    }
  }
  
  /**
   * Clear the current visualization
   */
  private clear(): void {
    if (this.atoms) {
      this.scene.remove(this.atoms)
      this.atoms.dispose()
      this.atoms = null
    }
    
    if (this.bonds) {
      this.scene.remove(this.bonds)
      this.bonds.dispose()
      this.bonds = null
    }
    
    this.polyhedra.forEach(p => {
      this.scene.remove(p)
      // Dispose of polyhedron geometries and materials
    })
    this.polyhedra = []
  }
  
  /**
   * Dispose of all resources
   */
  public dispose(): void {
    this.clear()
  }
}