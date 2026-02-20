import * as THREE from 'three';

export interface DragStateManagerOptions {
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    camera: THREE.Camera;
    container: HTMLElement;
    controls: { enabled: boolean };
}

export class DragStateManager {
    private scene: THREE.Scene;
    private renderer: THREE.WebGLRenderer;
    private camera: THREE.Camera;
    private container: HTMLElement;
    private controls: { enabled: boolean };

    private mousePos: THREE.Vector2;
    private raycaster: THREE.Raycaster;
    private grabDistance: number;
    private active: boolean;
    private mouseDown: boolean;

    // Currently dragged object
    physicsObject: THREE.Object3D | null;

    // For force calculation
    localHit: THREE.Vector3;
    worldHit: THREE.Vector3;
    currentWorld: THREE.Vector3;
    offset: THREE.Vector3;

    // Debug arrow for force visualization
    private arrow: THREE.Group;
    private arrowShaft: THREE.Mesh;
    private arrowHead: THREE.Mesh;

    constructor(options: DragStateManagerOptions) {
        this.scene = options.scene;
        this.renderer = options.renderer;
        this.camera = options.camera;
        this.container = options.container;
        this.controls = options.controls;

        this.mousePos = new THREE.Vector2();
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Line!.threshold = 0.1;

        this.grabDistance = 0.0;
        this.active = false;
        this.mouseDown = false;
        this.physicsObject = null;

        this.localHit = new THREE.Vector3();
        this.worldHit = new THREE.Vector3();
        this.currentWorld = new THREE.Vector3();
        this.offset = new THREE.Vector3();

        // Force vector visualization - custom thick 3D arrow
        this.arrow = new THREE.Group();

        // Create thick cylinder shaft (radius, height, radialSegments)
        const shaftGeometry = new THREE.CylinderGeometry(0.008, 0.008, 1);
        const arrowMaterial = new THREE.MeshStandardMaterial({
            color: 0xff6347,
            transparent: true,
            opacity: 0.5,
            metalness: 0,
            roughness: 0.2
        });
        this.arrowShaft = new THREE.Mesh(shaftGeometry, arrowMaterial);
        // Cylinder is centered, so translate it up by half its height
        this.arrowShaft.position.y = 0.5;

        // Create cone head (radius, height, radialSegments)
        const headGeometry = new THREE.ConeGeometry(0.03, 0.10);
        this.arrowHead = new THREE.Mesh(headGeometry, arrowMaterial);
        // Position cone at the top of the shaft
        this.arrowHead.position.y = 1;

        this.arrow.add(this.arrowShaft);
        this.arrow.add(this.arrowHead);
        this.arrow.visible = false;
        this.scene.add(this.arrow);

        // Register event listeners
        this.container.addEventListener('pointerdown', this.onPointer, true);
        document.addEventListener('pointermove', this.onPointer, true);
        document.addEventListener('pointerup', this.onPointer, true);
    }

    private updateRaycaster(x: number, y: number): void {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mousePos.x = ((x - rect.left) / rect.width) * 2 - 1;
        this.mousePos.y = -((y - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mousePos, this.camera);
    }

    private start(x: number, y: number): void {
        this.physicsObject = null;
        this.updateRaycaster(x, y);

        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        for (const intersect of intersects) {
            const obj = intersect.object;
            // Only objects with bodyID > 0 are draggable (excludes world/plane)
            if ('bodyID' in obj && typeof obj.bodyID === 'number' && obj.bodyID > 0) {
                this.physicsObject = obj;
                this.grabDistance = intersect.distance;

                const hit = this.raycaster.ray.origin
                    .clone()
                    .addScaledVector(this.raycaster.ray.direction, this.grabDistance);

                this.localHit.copy(obj.worldToLocal(hit.clone()));
                this.worldHit.copy(hit);
                this.currentWorld.copy(hit);
                // Reset offset
                this.offset.set(0, 0, 0);

                this.arrow.position.copy(hit);
                // Don't show arrow yet until drag actually starts
                this.arrow.visible = false;

                this.active = true;
                this.controls.enabled = false; // Disable OrbitControls

                break;
            }
        }

        // If no object with bodyID is found, controls remain enabled
    }

    private move(x: number, y: number): void {
        if (!this.active) {
            return;
        }

        this.updateRaycaster(x, y);
        const hit = this.raycaster.ray.origin
            .clone()
            .addScaledVector(this.raycaster.ray.direction, this.grabDistance);

        this.currentWorld.copy(hit);
        this.update();
    }

    update(): void {
        if (!this.physicsObject || !this.active) {
            return;
        }

        // Recalculate world position of physicsObject
        this.worldHit.copy(this.localHit);
        this.physicsObject.localToWorld(this.worldHit);

        // Offset from target position (force direction and magnitude)
        this.offset.copy(this.currentWorld).sub(this.worldHit);

        // Update debug arrow
        this.arrow.position.copy(this.worldHit);
        const length = this.offset.length();
        if (length > 0.001) {
            // Show arrow only when there's actual drag movement
            this.arrow.visible = true;

            // Set direction by making the arrow point toward the offset
            const direction = this.offset.clone().normalize();
            this.arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

            // Scale shaft to match length (subtract head height)
            const headHeight = 0.10; // Must match cone geometry height
            const shaftLength = Math.max(0.01, length - headHeight);
            this.arrowShaft.scale.y = shaftLength;
            this.arrowShaft.position.y = shaftLength / 2;

            // Position head so its base aligns with shaft end (cone is centered, so offset by half height)
            this.arrowHead.position.y = shaftLength + headHeight / 2;
        } else {
            // Hide arrow when offset is too small
            this.arrow.visible = false;
        }
    }

    end(): void {
        this.physicsObject = null;
        this.active = false;
        this.controls.enabled = true; // Re-enable OrbitControls
        this.arrow.visible = false;
        this.mouseDown = false;
    }

    private onPointer = (evt: PointerEvent): void => {
        if (evt.type === 'pointerdown') {
            this.start(evt.clientX, evt.clientY);
            this.mouseDown = true;
        } else if (evt.type === 'pointermove' && this.mouseDown) {
            this.move(evt.clientX, evt.clientY);
        } else if (evt.type === 'pointerup') {
            this.end();
        }
    };

    dispose(): void {
        this.container.removeEventListener('pointerdown', this.onPointer, true);
        document.removeEventListener('pointermove', this.onPointer, true);
        document.removeEventListener('pointerup', this.onPointer, true);

        if (this.arrow.parent) {
            this.scene.remove(this.arrow);
        }
        // Dispose geometries and materials
        this.arrowShaft.geometry.dispose();
        this.arrowHead.geometry.dispose();
        if (this.arrowShaft.material) {
            (this.arrowShaft.material as THREE.Material).dispose();
        }
    }
}
