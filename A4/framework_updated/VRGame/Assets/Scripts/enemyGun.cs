using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;


public class enemyGun : MonoBehaviour {

    public GameObject end, start; // The gun start and end point
    public GameObject gun;
    public Animator animator;
    
    public GameObject spine;
    public GameObject handMag;
    public GameObject gunMag;

    float gunShotTime = 0.2f;  // Time between shots, for 5 shots per second
    Quaternion previousRotation;
    public float health = 100;
    public bool isDead  = false;

    public static bool leftHanded { get; private set; }

    public GameObject bulletHole;
    public GameObject muzzleFlash;
    public GameObject shotSound;

    public GameObject player; // Reference to the player

    public float bulletDamage = 20.0f; // Damage per bullet
    public float shootingRange = 10.0f; // Range within which the enemy can shoot
    public float accuracyRange = 5.0f; // Randomization range for accuracy

    private float lastShotTime = 0.0f;

    public GameObject playerGun; // Reference to the player's gun

    public enemy Enemy; // Reference to the enemy

    public float dropOffset = 4.0f;

    // Use this for initialization
    void Start() {

    }

    // Update is called once per frame
    void Update() {

        if (!isDead)
        {
            Gun playerGunScript = playerGun.GetComponent<Gun>();
            if (playerGunScript.isDead) {
                return;
            }

            if (ShouldShootPlayer()) {
                ShootAtPlayer();
            }
        }
    }

    bool ShouldShootPlayer() {
        float distanceToPlayer = Vector3.Distance(end.transform.position, player.transform.position);
        bool isPlayerInRange = distanceToPlayer <= shootingRange;
        return isPlayerInRange && Time.time > lastShotTime + gunShotTime && animator.GetBool("dist_at_10");
    }

    void ShootAtPlayer() {
        lastShotTime = Time.time;

        Vector3 shootingDirection = (player.transform.position - end.transform.position).normalized;
        shootingDirection = Quaternion.Euler(Random.Range(-accuracyRange, accuracyRange), Random.Range(-accuracyRange, accuracyRange), 0) * shootingDirection;

        RaycastHit rayHit;
        if (Physics.Raycast(end.transform.position, shootingDirection, out rayHit, shootingRange)) {
            if (rayHit.collider.gameObject == player) {
                // 20% chance to hit
                if (Random.value <= 0.2f) {
                    Gun playerGunScript = playerGun.GetComponent<Gun>();
                    if (playerGunScript != null) {
                        playerGunScript.Being_shot(bulletDamage); // This replaces direct health manipulation
                    }
                }
                addEffects();
            }
        }
    }

    void addEffects() {
        RaycastHit rayHit;
        int layerMask = 1<<8;
        layerMask = ~layerMask;
        if(Physics.Raycast(end.transform.position, (end.transform.position - start.transform.position).normalized, out rayHit, 100.0f, layerMask))
        {
            GameObject bulletHoleObject = Instantiate(bulletHole, rayHit.point + rayHit.collider.transform.up*0.01f, rayHit.collider.transform.rotation);
            Destroy(bulletHoleObject, 2.0f);
        }

        GameObject muzzleFlashObject = Instantiate(muzzleFlash, end.transform.position, end.transform.rotation);
        muzzleFlashObject.GetComponent<ParticleSystem>().Play();
        Destroy(muzzleFlashObject, 1.0f);

        Destroy((GameObject) Instantiate(shotSound, transform.position, transform.rotation), 1.0f);
    }

    public void Being_shot(float damage) // getting hit from enemy
    {
        // Debug.Log("enemy being shot");

        animator.SetBool("player_detect", true);

        if (isDead) return;

        health -= damage;

        if (health <= 0) 
        {
            Die();
        }

        // Debug.Log("current enemy health: " + health);
    }

    void Die() {
        isDead = true;
        animator.SetTrigger("die");
        Enemy.isDead = true;


        // Disable the original gun
        if (gun != null) 
        {
            gun.SetActive(false);
        }

        // drop gun logic
        Vector3 dropPosition = transform.position + transform.forward * dropOffset + new Vector3(0, 2f, 0); // Ensure it's above the floor
        Quaternion dropRotation = Quaternion.Euler(0, transform.eulerAngles.y, 0);
        GameObject droppedGun = Instantiate(gun, dropPosition, dropRotation);
        Rigidbody gunRigidbody = droppedGun.AddComponent<Rigidbody>();
        BoxCollider gunCollider = droppedGun.AddComponent<BoxCollider>();

        // Disable or remove the gun from the enemy's hand
        droppedGun.SetActive(true);
    }
}
