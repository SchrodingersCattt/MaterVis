import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { CrystalRenderer } from './renderer'
import { CrystalStructure } from './crystal'
import { useStore } from './store'

export class Viewer {
  private container: HTMLDivElement
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private clock: THREE.Clock
  private crystalRenderer: CrystalRenderer

  constructor(container: HTMLDivElement) {
    this.container = container
    
    // Create Three.js components
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0xf0f0f0)
    
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000)
    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.clock = new THREE.Clock()
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    
    // Create crystal renderer
    this.crystalRenderer = new CrystalRenderer(this.scene)
    
    // Initialize the viewer
    this.init()
    
    // Start animation loop
    this.animate()
    
    // Subscribe to state changes
    useStore.subscribe(this.handleStateChange.bind(this))
  }
  
  private init(): void {
    // Set up renderer
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.container.appendChild(this.renderer.domElement)
    
    // Configure controls
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    
    // Add lighting for plastic material appearance
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    this.scene.add(ambientLight)
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight1.position.set(1, 1, 1)
    this.scene.add(directionalLight1)
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4)
    directionalLight2.position.set(-1, -1, -1)
    this.scene.add(directionalLight2)
    
    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3)
    this.scene.add(hemisphereLight)
    
    // Position camera
    this.camera.position.set(0, 0, 10)
    
    // Handle window resize
    window.addEventListener('resize', this.onWindowResize.bind(this), false)
  }
  
  private onWindowResize(): void {
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight)
  }
  
  private animate(): void {
    requestAnimationFrame(this.animate.bind(this))
    
    const delta = this.clock.getDelta()
    
    // Update controls
    this.controls.update()
    
    // Render scene
    this.renderer.render(this.scene, this.camera)
  }
  
  public loadCrystal(crystal: CrystalStructure): void {
    useStore.getState().setCrystal(crystal)
    this.crystalRenderer.loadCrystal(crystal)
    
    // Center camera on crystal
    this.centerView()
  }
  
  private centerView(): void {
    // Simple centering - in a full implementation we would compute the bounding box
    this.controls.target.set(0, 0, 0)
    this.camera.position.set(0, 0, 10)
    this.controls.update()
  }
  
  private handleStateChange(): void {
    // Update visualization based on state changes
    this.crystalRenderer.update()
  }
  
  public dispose(): void {
    // Clean up resources
    this.controls.dispose()
    this.renderer.dispose()
    this.crystalRenderer.dispose()
  }
}