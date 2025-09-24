/**
 * UI Controls for the physics simulation
 */

class SimulationControls {
    constructor(physics, renderer) {
        this.physics = physics;
        this.renderer = renderer;
        this.setupControls();
        this.bindEvents();
    }
    
    setupControls() {
        // Get control elements
        this.playPauseBtn = document.getElementById('playPause');
        this.cleanupBtn = document.getElementById('cleanup');
        this.speedSlider = document.getElementById('speed');
        this.speedValue = document.getElementById('speedValue');
        this.timeDisplay = document.getElementById('timeDisplay');
        this.fieldStrengthSlider = document.getElementById('fieldStrength');
        this.fieldValue = document.getElementById('fieldValue');
        this.showOriginalCheckbox = document.getElementById('showOriginal');
        this.showElectronCheckbox = document.getElementById('showElectron');
        
        // Field value displays
        this.originalValue = document.getElementById('originalValue');
        this.electronValue = document.getElementById('electronValue');
        this.combinedValue = document.getElementById('combinedValue');
        
        // Performance displays
        this.fpsValue = document.getElementById('fpsValue');
        this.memoryValue = document.getElementById('memoryValue');
        
        // Performance monitoring
        this.lastTime = performance.now();
        this.frameCount = 0;
    }
    
    bindEvents() {
        // Play/Pause button
        this.playPauseBtn.addEventListener('click', () => {
            const isRunning = this.physics.toggle();
            this.playPauseBtn.textContent = isRunning ? 'Pause' : 'Play';
        });
        
        // Cleanup button
        this.cleanupBtn.addEventListener('click', () => {
            console.log('🧹 Manual cleanup triggered');
            if (window.gc) {
                window.gc();
                console.log('✅ Garbage collection completed');
            } else {
                console.log('⚠️ Garbage collection not available');
            }
        });
        
        // Speed control
        this.speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.physics.setSpeed(speed);
            this.speedValue.textContent = speed.toFixed(1) + 'x';
            
            // Warning for high speeds
            if (speed > 3) {
                this.speedValue.style.color = '#ff6b6b';
                this.speedValue.textContent += ' (High!)';
            } else {
                this.speedValue.style.color = '#e0e0e0';
            }
        });
        
        // Field strength control
        this.fieldStrengthSlider.addEventListener('input', (e) => {
            const strength = parseFloat(e.target.value);
            this.physics.setFieldStrength(strength);
            this.fieldValue.textContent = strength.toFixed(1);
        });
        
        // Show/hide controls
        this.showOriginalCheckbox.addEventListener('change', (e) => {
            this.updateFieldVisibility();
        });
        
        this.showElectronCheckbox.addEventListener('change', (e) => {
            this.updateFieldVisibility();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.renderer.resize();
        });
    }
    
    updateFieldVisibility() {
        const showOriginal = this.showOriginalCheckbox.checked;
        const showElectron = this.showElectronCheckbox.checked;
        
        // Update arrow visibility immediately
        this.renderer.originalFieldArrows.visible = showOriginal;
        this.renderer.electronFieldArrows.visible = showElectron;
        
        // Force a re-render to show changes
        this.renderer.render();
    }
    
    updateTimeDisplay() {
        this.timeDisplay.textContent = this.physics.time.toFixed(2) + 's';
    }
    
    updateFieldValues(fieldData) {
        if (fieldData && fieldData.length > 0) {
            // Get a sample field value (middle of the data)
            const sampleField = fieldData[Math.floor(fieldData.length / 2)];
            if (sampleField) {
                this.originalValue.textContent = sampleField.field.original.toFixed(3);
                this.electronValue.textContent = sampleField.field.electron.toFixed(3);
                this.combinedValue.textContent = sampleField.field.combined.toFixed(3);
            }
        }
    }
    
    update() {
        this.updateTimeDisplay();
        this.updatePerformance();
    }
    
    updatePerformance() {
        this.frameCount++;
        const currentTime = performance.now();
        
        if (currentTime - this.lastTime >= 1000) { // Update every second
            const fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.fpsValue.textContent = fps;
            
            // Memory usage (if available)
            if (performance.memory) {
                const memoryMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
                this.memoryValue.textContent = memoryMB + 'MB';
            }
            
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }
}
