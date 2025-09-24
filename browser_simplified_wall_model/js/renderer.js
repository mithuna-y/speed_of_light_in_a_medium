/**
 * Three.js renderer for 3D field visualization
 */

class FieldRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        
        this.setupRenderer();
        this.setupScene();
        this.setupLights();
        this.setupObjects();
        
        // Animation state
        this.animationId = null;
        this.isAnimating = false;
    }
    
    setupRenderer() {
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setClearColor(0x0a0a0a, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    
    setupScene() {
        // Camera position
        this.camera.position.set(15, 15, 15);
        this.camera.lookAt(0, 0, 0);
        
        // Add grid helper
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);
    }
    
    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Point light for better illumination
        const pointLight = new THREE.PointLight(0x4a90e2, 0.5, 50);
        pointLight.position.set(0, 10, 0);
        this.scene.add(pointLight);
    }
    
    setupObjects() {
        // Field arrows (will be updated dynamically)
        this.fieldArrows = new THREE.Group();
        this.scene.add(this.fieldArrows);
        
        // Electron particles
        this.electronGroup = new THREE.Group();
        this.scene.add(this.electronGroup);
        
        // Original field arrows
        this.originalFieldArrows = new THREE.Group();
        this.scene.add(this.originalFieldArrows);
        
        // Electron field arrows
        this.electronFieldArrows = new THREE.Group();
        this.scene.add(this.electronFieldArrows);
    }
    
    createArrow(position, direction, color, scale = 1) {
        const arrowGeometry = new THREE.ConeGeometry(0.1, 0.5, 8);
        const arrowMaterial = new THREE.MeshLambertMaterial({ color: color });
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        
        // Position the arrow
        arrow.position.copy(position);
        
        // Orient the arrow
        const directionVector = new THREE.Vector3().copy(direction).normalize();
        arrow.lookAt(
            position.x + directionVector.x,
            position.y + directionVector.y,
            position.z + directionVector.z
        );
        
        // Scale based on field strength
        const fieldStrength = direction.length();
        arrow.scale.setScalar(Math.max(0.1, Math.min(2, fieldStrength * scale)));
        
        return arrow;
    }
    
    createElectron(position) {
        const electronGeometry = new THREE.SphereGeometry(0.2, 8, 6);
        const electronMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x4a90e2,
            emissive: 0x1a4a7a,
            emissiveIntensity: 0.3
        });
        const electron = new THREE.Mesh(electronGeometry, electronMaterial);
        
        electron.position.copy(position);
        electron.castShadow = true;
        
        return electron;
    }
    
    disposeGroup(group) {
        // Properly dispose of all objects in a group to prevent memory leaks
        while (group.children.length > 0) {
            const child = group.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            group.remove(child);
        }
    }
    
    updateFieldArrows(fieldData, showOriginal = true, showElectron = true) {
        // Properly dispose of existing arrows to prevent memory leaks
        this.disposeGroup(this.fieldArrows);
        this.disposeGroup(this.originalFieldArrows);
        this.disposeGroup(this.electronFieldArrows);
        
        // Clear existing arrows
        this.fieldArrows.clear();
        this.originalFieldArrows.clear();
        this.electronFieldArrows.clear();
        
        // Find maximum field strength for scaling
        let maxOriginal = 0;
        let maxElectron = 0;
        let maxCombined = 0;
        
        fieldData.forEach(data => {
            maxOriginal = Math.max(maxOriginal, Math.abs(data.field.original));
            maxElectron = Math.max(maxElectron, Math.abs(data.field.electron));
            maxCombined = Math.max(maxCombined, Math.abs(data.field.combined));
        });
        
        // Scale factors to make fields visible
        const originalScale = maxOriginal > 0 ? 2.0 / maxOriginal : 1;
        const electronScale = maxElectron > 0 ? 2.0 / maxElectron : 1;
        const combinedScale = maxCombined > 0 ? 1.0 / maxCombined : 1;
        
        fieldData.forEach(data => {
            const { position, field } = data;
            const pos = new THREE.Vector3(position.x, position.y, position.z);
            
            // Original field arrows (red) - much lower threshold
            if (showOriginal && Math.abs(field.original) > 0.001) {
                const originalDirection = new THREE.Vector3(0, 0, field.original * originalScale);
                const originalArrow = this.createArrow(pos, originalDirection, 0xff4444, 1.0);
                this.originalFieldArrows.add(originalArrow);
            }
            
            // Electron field arrows (green) - much lower threshold
            if (showElectron && Math.abs(field.electron) > 0.001) {
                const electronDirection = new THREE.Vector3(0, 0, field.electron * electronScale);
                const electronArrow = this.createArrow(pos, electronDirection, 0x44ff44, 1.0);
                this.electronFieldArrows.add(electronArrow);
            }
            
            // Combined field arrows (magenta) - much lower threshold
            if (Math.abs(field.combined) > 0.001) {
                const combinedDirection = new THREE.Vector3(0, 0, field.combined * combinedScale);
                const combinedArrow = this.createArrow(pos, combinedDirection, 0xff44ff, 0.5);
                this.fieldArrows.add(combinedArrow);
            }
        });
        
        // Set visibility of arrow groups
        this.originalFieldArrows.visible = showOriginal;
        this.electronFieldArrows.visible = showElectron;
    }
    
    updateElectrons(electronData) {
        // Properly dispose of existing electrons to prevent memory leaks
        this.disposeGroup(this.electronGroup);
        
        // Clear existing electrons
        this.electronGroup.clear();
        
        electronData.forEach(data => {
            const electron = this.createElectron(data.position);
            this.electronGroup.add(electron);
        });
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        this.render();
    }
    
    startAnimation() {
        if (!this.isAnimating) {
            this.isAnimating = true;
            this.animate();
        }
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.isAnimating = false;
    }
    
    resize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        this.stopAnimation();
        this.renderer.dispose();
    }
}
