import { create } from 'zustand'
import { CrystalStructure, StyleSettings, BondingSettings } from './crystal'

interface ViewState {
  // Crystal structure data
  crystal: CrystalStructure | null
  setCrystal: (crystal: CrystalStructure) => void
  
  // Visualization settings
  style: StyleSettings
  setStyle: (style: StyleSettings) => void
  
  // Bonding settings
  bonding: BondingSettings
  setBonding: (bonding: BondingSettings) => void
  
  // UI state
  showBonds: boolean
  setShowBonds: (show: boolean) => void
  
  showPolyhedra: boolean
  setShowPolyhedra: (show: boolean) => void
  
  showAxes: boolean
  setShowAxes: (show: boolean) => void
  
  // Disorder settings
  disorderMode: 'average' | 'sample'
  setDisorderMode: (mode: 'average' | 'sample') => void
  
  // Supercell settings
  supercell: [number, number, number]
  setSupercell: (size: [number, number, number]) => void
}

export const useStore = create<ViewState>()((set) => ({
  crystal: null,
  setCrystal: (crystal) => set({ crystal }),
  
  style: {
    colors: {},
    bonding: {
      mode: 'covalent',
      f: 1.1,
      min: 0.7,
      max: 3.2
    }
  },
  setStyle: (style) => set({ style }),
  
  bonding: {
    mode: 'covalent',
    f: 1.1,
    min: 0.7,
    max: 3.2
  },
  setBonding: (bonding) => set({ bonding }),
  
  showBonds: true,
  setShowBonds: (show) => set({ showBonds: show }),
  
  showPolyhedra: false,
  setShowPolyhedra: (show) => set({ showPolyhedra: show }),
  
  showAxes: true,
  setShowAxes: (show) => set({ showAxes: show }),
  
  disorderMode: 'average',
  setDisorderMode: (mode) => set({ disorderMode: mode }),
  
  supercell: [1, 1, 1],
  setSupercell: (size) => set({ supercell: size })
}))