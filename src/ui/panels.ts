/**
 * UI Panel system for the crystal visualizer
 */

export class UIPanel {
  protected element: HTMLDivElement
  
  constructor(title: string) {
    this.element = document.createElement('div')
    this.element.className = 'ui-panel'
    
    const header = document.createElement('div')
    header.className = 'panel-header'
    header.textContent = title
    this.element.appendChild(header)
  }
  
  getElement(): HTMLDivElement {
    return this.element
  }
  
  addControl(control: HTMLElement): void {
    this.element.appendChild(control)
  }
}

export class ControlPanel extends UIPanel {
  private content: HTMLDivElement
  
  constructor(title: string) {
    super(title)
    
    this.content = document.createElement('div')
    this.content.className = 'panel-content'
    this.element.appendChild(this.content)
  }
  
  addControl(control: HTMLElement): void {
    this.content.appendChild(control)
  }
}

export function createCheckbox(label: string, checked: boolean = false, onChange: (checked: boolean) => void = () => {}): HTMLLabelElement {
  const labelEl = document.createElement('label')
  labelEl.className = 'control-checkbox'
  
  const checkbox = document.createElement('input')
  checkbox.type = 'checkbox'
  checkbox.checked = checked
  checkbox.addEventListener('change', (e) => {
    onChange((e.target as HTMLInputElement).checked)
  })
  
  labelEl.appendChild(checkbox)
  labelEl.appendChild(document.createTextNode(label))
  
  return labelEl
}

export function createSlider(
  label: string, 
  min: number, 
  max: number, 
  value: number, 
  step: number = 0.1,
  onChange: (value: number) => void = () => {}
): HTMLDivElement {
  const container = document.createElement('div')
  container.className = 'control-slider'
  
  const labelEl = document.createElement('label')
  labelEl.textContent = label
  container.appendChild(labelEl)
  
  const sliderContainer = document.createElement('div')
  sliderContainer.className = 'slider-container'
  
  const slider = document.createElement('input')
  slider.type = 'range'
  slider.min = min.toString()
  slider.max = max.toString()
  slider.step = step.toString()
  slider.value = value.toString()
  
  const valueDisplay = document.createElement('span')
  valueDisplay.className = 'slider-value'
  valueDisplay.textContent = value.toString()
  
  slider.addEventListener('input', (e) => {
    const val = parseFloat((e.target as HTMLInputElement).value)
    valueDisplay.textContent = val.toFixed(2)
    onChange(val)
  })
  
  sliderContainer.appendChild(slider)
  sliderContainer.appendChild(valueDisplay)
  container.appendChild(sliderContainer)
  
  return container
}

export function createButton(label: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement('button')
  button.className = 'control-button'
  button.textContent = label
  button.addEventListener('click', onClick)
  return button
}