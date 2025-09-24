/**
 * Main application for the Simplified Wall Model browser simulation
 */

class PhysicsApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.physics = new PhysicsSimulation();
        this.renderer = new FieldRenderer(this.canvas);
        this.controls = new SimulationControls(this.physics, this.renderer);
        
        this.animationId = null;
        this.isRunning = false;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Physics Simulation...');
        
        // Start the renderer animation loop
        this.renderer.startAnimation();
        
        // Start the simulation
        this.start();
        
        console.log('✅ Physics Simulation Ready!');
    }
    
    start() {
        this.isRunning = true;
        this.physics.start();
        this.animate();
    }
    
    pause() {
        this.isRunning = false;
        this.physics.pause();
    }
    
    animate() {
        if (!this.isRunning) return;
        
        // Frame rate limiting - cap at 60 FPS
        const currentTime = performance.now();
        if (currentTime - this.lastFrameTime < 16.67) { // 60 FPS = 16.67ms per frame
            this.animationId = requestAnimationFrame(() => this.animate());
            return;
        }
        this.lastFrameTime = currentTime;
        
        // Update physics
        this.physics.step();
        
        // Only update renderer every few frames to reduce memory pressure
        if (this.frameCount % 3 === 0) {
            // Get updated data
            const fieldData = this.physics.getFieldData();
            const electronData = this.physics.getElectronData();
            
            // Update renderer
            this.renderer.updateFieldArrows(
                fieldData, 
                this.controls.showOriginalCheckbox.checked,
                this.controls.showElectronCheckbox.checked
            );
            this.renderer.updateElectrons(electronData);
            
            // Update controls
            this.controls.updateFieldValues(fieldData);
        }
        
        // Update time display every frame
        this.controls.updateTimeDisplay();
        
        // Debug: Log field values occasionally (reduced frequency)
        if (this.frameCount % 100 === 0) {
            const fieldData = this.physics.getFieldData();
            const sampleField = fieldData[Math.floor(fieldData.length / 2)];
            if (sampleField) {
                console.log(`Field values at t=${this.physics.time.toFixed(2)}:`, {
                    original: sampleField.field.original.toFixed(4),
                    electron: sampleField.field.electron.toFixed(4),
                    combined: sampleField.field.combined.toFixed(4)
                });
            }
        }
        
        this.frameCount++;
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    resize() {
        this.renderer.resize();
    }
    
    dispose() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.renderer.dispose();
        
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌐 Loading Physics Simulation...');
    
    // Create the main application
    window.physicsApp = new PhysicsApp();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        window.physicsApp.resize();
    });
    
    // Handle page unload
    window.addEventListener('beforeunload', () => {
        window.physicsApp.dispose();
    });
    
    console.log('🎯 Physics Simulation Loaded Successfully!');
});

// Add some helpful console commands for debugging
window.addEventListener('load', () => {
    console.log('🔧 Debug Commands Available:');
    console.log('  physicsApp.physics - Access physics simulation');
    console.log('  physicsApp.renderer - Access 3D renderer');
    console.log('  physicsApp.controls - Access UI controls');
    console.log('  physicsApp.start() - Start simulation');
    console.log('  physicsApp.pause() - Pause simulation');
});
