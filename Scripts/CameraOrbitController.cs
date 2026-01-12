using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Camera orbit controller that mimics Unity Scene View controls.
/// WASD for movement, QE for up/down, Shift for faster movement.
/// Middle mouse or Alt+Left mouse to orbit, mouse wheel to zoom.
/// Camera position resets when entering play mode.
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraOrbitController : MonoBehaviour
{
    [Header("Movement Settings")]
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float fastMoveSpeed = 15f;
    [SerializeField] private float smoothFactor = 0.95f; // Smooth interpolation factor (0-1, higher = smoother)
    
    [Header("Orbit Settings")]
    [SerializeField] private Transform orbitTarget;
    [SerializeField] private float orbitDistance = 10f;
    [SerializeField] private float minDistance = 1f;
    [SerializeField] private float maxDistance = 50f;
    [SerializeField] private float orbitSpeed = 2f;
    
    [Header("Mouse Settings")]
    [SerializeField] private bool invertY = false;
    [SerializeField] private float mouseSensitivity = 2f;
    [SerializeField] private float scrollSensitivity = 2f;
    
    private Camera cam;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private Vector2 rotation;
    
    // Smooth movement targets
    private Vector3 targetPosition;
    private Vector2 targetRotation;
    
    private void Awake()
    {
        cam = GetComponent<Camera>();
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        
        // Calculate initial rotation angles
        rotation = new Vector2(transform.eulerAngles.y, transform.eulerAngles.x);
        targetRotation = rotation;
        targetPosition = transform.position;
    }
    
    private void OnEnable()
    {
        // Reset to initial position when entering play mode
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        rotation = new Vector2(transform.eulerAngles.y, transform.eulerAngles.x);
        targetRotation = rotation;
        targetPosition = initialPosition;
        
        // If we have an orbit target, calculate distance
        if (orbitTarget != null)
        {
            orbitDistance = Vector3.Distance(transform.position, orbitTarget.position);
            orbitDistance = Mathf.Clamp(orbitDistance, minDistance, maxDistance);
        }
    }
    
    private void Update()
    {
        // Check for orbit input (middle mouse or Alt+Left mouse, like Scene view)
        bool middleMouseDown = Mouse.current != null && Mouse.current.middleButton.isPressed;
        bool altPressed = Keyboard.current != null && (Keyboard.current.leftAltKey.isPressed || Keyboard.current.rightAltKey.isPressed);
        bool leftMouseDown = Mouse.current != null && Mouse.current.leftButton.isPressed;
        bool rightMouseDown = Mouse.current != null && Mouse.current.rightButton.isPressed;
        
        bool isOrbiting = (middleMouseDown || (altPressed && leftMouseDown)) && orbitTarget != null;
        bool isFreeLooking = rightMouseDown; // Right mouse for free look (first-person style)
        
        if (isOrbiting)
        {
            HandleMouseOrbit();
        }
        else if (isFreeLooking)
        {
            HandleFreeLook();
        }
        else
        {
            HandleKeyboardMovement();
        }
        
        // Handle mouse wheel zoom
        if (orbitTarget != null && Mouse.current != null)
        {
            Vector2 scroll = Mouse.current.scroll.ReadValue();
            if (scroll.y != 0)
            {
                orbitDistance -= scroll.y * scrollSensitivity * 0.01f;
                orbitDistance = Mathf.Clamp(orbitDistance, minDistance, maxDistance);
                
                // Update position to maintain orbit
                Quaternion rot = Quaternion.Euler(targetRotation.y, targetRotation.x, 0f);
                Vector3 dir = rot * Vector3.back;
                targetPosition = orbitTarget.position + dir * orbitDistance;
            }
        }
        
        // Apply smooth interpolation
        ApplySmoothMovement();
    }
    
    private void ApplySmoothMovement()
    {
        // Smooth position interpolation (similar to Swift camera)
        float factor = 1f - Mathf.Pow(smoothFactor, Time.deltaTime * 60f);
        transform.position = Vector3.Lerp(transform.position, targetPosition, factor);
        
        // Apply rotation instantly (no smoothing)
        rotation = targetRotation;
        transform.rotation = Quaternion.Euler(rotation.y, rotation.x, 0f);
    }
    
    private void HandleKeyboardMovement()
    {
        if (Keyboard.current == null) return;
        
        // Check for fast movement (Shift)
        bool isFastMove = Keyboard.current.leftShiftKey.isPressed || Keyboard.current.rightShiftKey.isPressed;
        float currentSpeed = isFastMove ? fastMoveSpeed : moveSpeed;
        
        // Get movement direction relative to camera
        Vector3 forward = transform.forward;
        Vector3 right = transform.right;
        Vector3 up = Vector3.up;
        
        // Remove vertical component from forward/right for flat movement
        forward.y = 0f;
        right.y = 0f;
        forward.Normalize();
        right.Normalize();
        
        // Read WASD input
        Vector2 moveInput = Vector2.zero;
        if (Keyboard.current.wKey.isPressed) moveInput.y += 1f;
        if (Keyboard.current.sKey.isPressed) moveInput.y -= 1f;
        if (Keyboard.current.aKey.isPressed) moveInput.x -= 1f;
        if (Keyboard.current.dKey.isPressed) moveInput.x += 1f;
        
        // Read QE input for vertical movement
        float verticalInput = 0f;
        if (Keyboard.current.qKey.isPressed) verticalInput -= 1f;
        if (Keyboard.current.eKey.isPressed) verticalInput += 1f;
        
        // Calculate target movement
        Vector3 movement = (forward * moveInput.y + right * moveInput.x) * currentSpeed * Time.deltaTime;
        movement += up * verticalInput * currentSpeed * Time.deltaTime;
        
        // Update target position (will be smoothly interpolated)
        targetPosition += movement;
    }
    
    private void HandleMouseOrbit()
    {
        if (Mouse.current == null || orbitTarget == null) return;
        
        // Get mouse delta (not multiplied by deltaTime - mouse delta is already per-frame)
        Vector2 mouseDelta = Mouse.current.delta.ReadValue();
        
        if (mouseDelta.magnitude > 0.01f) // Only update if mouse moved
        {
            // Update target rotation (mouse delta is already frame-independent)
            targetRotation.x += mouseDelta.x * mouseSensitivity * 0.1f;
            targetRotation.y -= mouseDelta.y * mouseSensitivity * 0.1f * (invertY ? -1f : 1f);
            
            // Clamp vertical rotation
            targetRotation.y = Mathf.Clamp(targetRotation.y, -90f, 90f);
            
            // Calculate target position
            Quaternion rot = Quaternion.Euler(targetRotation.y, targetRotation.x, 0f);
            Vector3 dir = rot * Vector3.back;
            targetPosition = orbitTarget.position + dir * orbitDistance;
        }
    }
    
    private void HandleFreeLook()
    {
        if (Mouse.current == null) return;
        
        // Get mouse delta for free look rotation
        Vector2 mouseDelta = Mouse.current.delta.ReadValue();
        
        if (mouseDelta.magnitude > 0.01f)
        {
            // Update target rotation for free look
            targetRotation.x += mouseDelta.x * mouseSensitivity * 0.1f;
            targetRotation.y -= mouseDelta.y * mouseSensitivity * 0.1f * (invertY ? -1f : 1f);
            
            // Clamp vertical rotation
            targetRotation.y = Mathf.Clamp(targetRotation.y, -90f, 90f);
        }
        
        // Still allow keyboard movement during free look
        HandleKeyboardMovement();
    }
    
    // Public method to set orbit target
    public void SetOrbitTarget(Transform target)
    {
        orbitTarget = target;
        if (target != null)
        {
            // Calculate distance to target
            orbitDistance = Vector3.Distance(transform.position, target.position);
            orbitDistance = Mathf.Clamp(orbitDistance, minDistance, maxDistance);
        }
    }
}
