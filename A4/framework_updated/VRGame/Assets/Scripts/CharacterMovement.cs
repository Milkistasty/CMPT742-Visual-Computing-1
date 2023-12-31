using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterMovement : MonoBehaviour {
    public GameObject Camera;
    public GameObject CameraPlace;
    public GameObject cameraFinalPos;
    Animator animator;
    public GameObject spine;

    public bool isDead  = false;

    private Vector3 move_direction;
    private Vector2 mouse_movement;
    private Vector2 mouse_angles;
    private Vector2 mouse_y_limits = new Vector2(-50f, 50f);
    public float sensitivity = 5f;

    // Use this for initialization
    void Start () {
        Cursor.lockState = CursorLockMode.Locked;

        animator = GetComponent<Animator>();
        // Initializing animator values
        animator.SetFloat("walk_forward", 0.0f);
        animator.SetFloat("walk_backward", 0.0f);
        animator.SetFloat("walk_right", 0.0f);
        animator.SetFloat("walk_left", 0.0f);
    }

    // Update is called once per frame
    void Update () {
        // LockAndUnlockCursor();

        if(Cursor.lockState == CursorLockMode.Locked)
        {
            move_direction = new Vector3(Input.GetAxis("Horizontal"), 0f, Input.GetAxis("Vertical"));

            if (!isDead)
            {
                // The player should not move
                if (move_direction.magnitude < 0.1f)
                {
                    animator.SetFloat("walk_forward", -1f);
                    animator.SetFloat("walk_backward", -1f);
                    animator.SetFloat("walk_right", -1f);
                    animator.SetFloat("walk_left", -1f);
                    animator.SetFloat("animation_speed", 0.0f);
                }
                else // The player should move
                {
                    float forwardSpeed = move_direction.z;
                    if (forwardSpeed > 0) // making forward walking speed faster
                    {
                        forwardSpeed = forwardSpeed * 2;
                    }

                    // Running the correct animation
                    animator.SetFloat("walk_forward", forwardSpeed);
                    animator.SetFloat("walk_backward", -move_direction.z);
                    animator.SetFloat("walk_right", move_direction.x);
                    animator.SetFloat("walk_left", -move_direction.x);

                    // Setting animation running speed
                    animator.SetFloat("animation_speed", Mathf.Sqrt(Mathf.Pow(move_direction.x, 2f) + Mathf.Pow(forwardSpeed, 2f)));
                }   
            }
        }
    }

    private void LateUpdate()
    {
        // Get mouse movement (Mouse Y --> Rotation around x-axis)
        mouse_movement = new Vector2(Input.GetAxis("Mouse Y"), Input.GetAxis("Mouse X"));
        // Apply mouse sensitivity and invert up/down movement
        mouse_angles.x += mouse_movement.x * sensitivity * -1;
        mouse_angles.y += mouse_movement.y * sensitivity;

        // Limit up and down view
        mouse_angles.x = Mathf.Clamp(mouse_angles.x, mouse_y_limits.x, mouse_y_limits.y);

        if (!isDead)
        {
            // Rotating the player spine according mouse y (upper body & gun)
            spine.transform.localRotation = Quaternion.Euler(mouse_angles.x, 0f, 0f);
            // Rotating camera according to mouse y
            // Camera is not child of spine, because spine of character model is a little twisted due to holding gun
            // Otherwise the camera would rotate around z-axis while looking up and down (similar to gun)
            Camera.transform.localRotation = spine.transform.localRotation;
            // Rotating the complete body of the player mouse x (upper & lower body)
            transform.localRotation = Quaternion.Euler(0f, mouse_angles.y, 0f);
        }

        if (!isDead) // Moving camera with body
        {
            Camera.transform.position = CameraPlace.transform.position;
        }
        else // Moving camera to a fixed point when the player is killed
        {
            Camera.transform.position = cameraFinalPos.transform.position;
            Camera.transform.rotation = cameraFinalPos.transform.rotation;
            cameraFinalPos.SetActive(true);
        }
    }
}
