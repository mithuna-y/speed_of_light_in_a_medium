/**
 * Physics calculations for the simplified wall model
 * Ported from Python implementation
 */

class PhysicsSimulation {
    constructor() {
        // Constants from original Python code
        this.GRID_RANGE = 10;
        this.GRID_SIZE = 10;
        this.TIME_END = 10;
        this.TIME_STEPS = 112;
        this.DT = this.TIME_END / this.TIME_STEPS;
        this.K = 0.5;
        this.OMEGA = 3.123;
        this.c = this.OMEGA / this.K; // speed of light
        this.hookes_constant = 0.5;
        this.damping_constant = 1;
        this.sigma = 0.8;
        
        // Electron grid setup
        this.setupElectronGrid();
        this.setupFieldPoints();
        
        // Simulation state
        this.time = 0;
        this.currentStep = 0;
        this.accel_history = new Map();
        this.isRunning = false;
        this.speed = 1.0;
        this.fieldStrength = 1.0;
    }
    
    setupElectronGrid() {
        // Create electron positions in x-z plane (y=0)
        this.x_e = [];
        this.z_e = [];
        this.y_e = [];
        
        for (let i = 0; i < this.GRID_SIZE; i++) {
            const x = -this.GRID_RANGE + 2 * this.GRID_RANGE * i / (this.GRID_SIZE - 1);
            this.x_e.push(x);
            this.z_e.push(x); // Same pattern for z
            this.y_e.push(0); // All electrons at y=0
        }
        
        // Initialize electron state
        this.z_previous = [...this.z_e];
        this.z_velocity = new Array(this.GRID_SIZE).fill(0);
        this.z_velocity_previous = new Array(this.GRID_SIZE).fill(0);
    }
    
    setupFieldPoints() {
        // Field evaluation points - reduced for better performance
        this.x_f = [0]; // Single point at x=0
        this.z_f = [0]; // Single point at z=0
        this.y_f = [];
        
        const grad = 50; // Reduced from 123 to 50 for better performance
        for (let i = 0; i < grad; i++) {
            const y = -this.GRID_RANGE + (2 * this.GRID_RANGE / grad) * i;
            this.y_f.push(y);
        }
    }
    
    // Original electric field function
    originalElectricField(t, k, y, omega) {
        return this.gaussian(t, y);
    }
    
    gaussian(t, y) {
        const coefficient = 10.0 / (this.sigma * Math.sqrt(2 * Math.PI)); // Increased coefficient
        const exponential_term = Math.exp(-0.5 * Math.pow((y + this.c * t - this.GRID_RANGE) / this.sigma, 2));
        return coefficient * exponential_term;
    }
    
    electronFieldContribution(x_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_scalar) {
        const r = Math.sqrt(Math.pow(x_e - x_p, 2) + Math.pow(y_p, 2) + Math.pow(z_e - z_p, 2));
        
        // Use a more generous threshold that's independent of speed scaling
        const threshold = this.DT * 2.5; // Make threshold more generous
        if (Math.abs((t_p - t_e) - r / this.c) < threshold) {
            // accel_scalar is now passed directly
            const contrib = -Math.sqrt(Math.pow(x_e - x_p, 2) + Math.pow(y_p, 2)) / r * accel_scalar / r;
            return (1/2) * contrib;
        }
        return 0;
    }
    
    updateElectrons(scaledDT) {
        // Calculate acceleration for the first electron (simplified)
        const accel = this.originalElectricField(this.time, this.K, 0, this.OMEGA) - 
                     this.hookes_constant * (this.z_previous[0] - this.z_e[0]) - 
                     this.damping_constant * this.z_velocity[0];
        
        // Store acceleration in history against the current time
        this.accel_history.set(this.time, accel);
        
        // Update electron positions and velocities using scaled time step
        for (let i = 0; i < this.x_e.length; i++) {
            for (let j = 0; j < this.z_e.length; j++) {
                this.z_previous[j] += this.z_velocity[j] * scaledDT + 0.5 * accel * Math.pow(scaledDT, 2);
                this.z_velocity[j] = this.z_velocity_previous[j] + accel * scaledDT;
            }
        }
        
        this.z_velocity_previous = [...this.z_velocity];
    }
    
    calculateFieldAtPoint(x_p, y_p, z_p) {
        const ef_original = this.originalElectricField(this.time, this.K, y_p, this.OMEGA);
        let ef_due_to_electrons = 0;
        
        let contributionsFound = 0;
        // Iterate through all stored past accelerations
        for (const [t_e, accel_scalar] of this.accel_history.entries()) {
            // For each past acceleration, calculate its contribution from all electrons
            for (const x_e of this.x_e) {
                for (const z_e of this.z_e) {
                    const contribution = this.electronFieldContribution(
                        x_e, z_e, t_e, x_p, y_p, z_p, this.time, accel_scalar
                    );
                    ef_due_to_electrons += contribution;
                    if (Math.abs(contribution) > 0.0001) {
                        contributionsFound++;
                    }
                }
            }
        }
        
        // Debug: Log when electron field is very small
        if (this.time > 1 && contributionsFound === 0) {
            console.log(`No electron field contributions found at t=${this.time.toFixed(2)}, speed=${this.speed}, history_size=${this.accel_history.size}`);
        }
        
        return {
            original: ef_original * this.fieldStrength,
            electron: ef_due_to_electrons * this.fieldStrength,
            combined: (ef_original + ef_due_to_electrons) * this.fieldStrength
        };
    }
    
    getFieldData() {
        const fieldData = [];
        
        for (const x_p of this.x_f) {
            for (const y_p of this.y_f) {
                for (const z_p of this.z_f) {
                    const field = this.calculateFieldAtPoint(x_p, y_p, z_p);
                    fieldData.push({
                        position: { x: x_p, y: y_p, z: z_p },
                        field: field
                    });
                }
            }
        }
        
        return fieldData;
    }
    
    getElectronData() {
        const electronData = [];
        
        for (let i = 0; i < this.x_e.length; i++) {
            for (let j = 0; j < this.z_e.length; j++) {
                electronData.push({
                    position: { 
                        x: this.x_e[i], 
                        y: this.y_e[i], 
                        z: this.z_previous[j] 
                    }
                });
            }
        }
        
        return electronData;
    }
    
    step() {
        if (!this.isRunning) return;
        
        // Apply speed scaling to the time step
        const scaledDT = this.DT * this.speed;
        
        this.updateElectrons(scaledDT);
        this.time += scaledDT;
        this.currentStep++;
        
        if (this.time >= this.TIME_END) {
            this.reset();
        }
    }
    
    reset() {
        this.time = 0;
        this.currentStep = 0;
        this.accel_history.clear();
        this.z_previous = [...this.z_e];
        this.z_velocity = new Array(this.GRID_SIZE).fill(0);
        this.z_velocity_previous = new Array(this.GRID_SIZE).fill(0);
    }
    
    setSpeed(speed) {
        // Limit speed to prevent simulation from running too fast
        this.speed = Math.max(0.1, Math.min(5.0, speed));
    }
    
    setFieldStrength(strength) {
        this.fieldStrength = strength;
    }
    
    start() {
        this.isRunning = true;
    }
    
    pause() {
        this.isRunning = false;
    }
    
    toggle() {
        this.isRunning = !this.isRunning;
        return this.isRunning;
    }
}
