using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.SceneManagement;


public class Gun : MonoBehaviour {

    public GameObject end, start; // The gun start and end point
    public GameObject gun;
    public Animator animator;
    
    public GameObject spine;
    public GameObject handMag;
    public GameObject gunMag;

    float gunShotTime = 0.1f;
    float gunReloadTime = 1.0f;
    Quaternion previousRotation;
    public float health = 1000;
    public bool isDead  = false;
 

    public Text magBullets;
    public Text remainingBullets;

    int magBulletsVal = 30;
    int remainingBulletsVal = 90;
    int magSize = 30;
    public GameObject headMesh;
    public static bool leftHanded { get; private set; }

    public GameObject bulletHole;
    public GameObject muzzleFlash;
    public GameObject shotSound;

    public CharacterMovement characterMovement; // Reference to the CharacterMovement script

    public Text healthDisplay; 

    public GameObject enemygun;
    public GameObject Enemy;
    public float accuracyRange = 5.0f; // Randomization range for accuracy
    public float bulletDamage = 50.0f; // Damage per bullet
    private float lastShotTime = 0.0f;
    public float shootingRange = 100.0f; // Range within which the player can shoot

    // Use this for initialization
    void Start() {
        headMesh.GetComponent<SkinnedMeshRenderer>().enabled = false; // Hiding player character head to avoid bugs :)
    }

    // Update is called once per frame
    void Update() {

        // Cool down times
        if (gunShotTime >= 0.0f)
        {
            gunShotTime -= Time.deltaTime;
        }
        if (gunReloadTime >= 0.0f)
        {
            gunReloadTime -= Time.deltaTime;
        }


        if ((Input.GetMouseButtonDown(0) || Input.GetMouseButton(0)) && gunShotTime <= 0 && gunReloadTime <= 0.0f && magBulletsVal > 0 && !isDead)
        { 
            shotDetection(); // Should be completed

            addEffects(); // Should be completed

            animator.SetBool("fire", true);
            gunShotTime = 0.5f;
            
            // Instantiating the muzzle prefab and shot sound
            
            magBulletsVal = magBulletsVal - 1;
            if (magBulletsVal <= 0 && remainingBulletsVal > 0)
            {
                animator.SetBool("reloadAfterFire", true);
                gunReloadTime = 2.5f;
                Invoke("reloaded", 2.5f);
            }
        }
        else
        {
            animator.SetBool("fire", false);
        }

        if ((Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.R)) && gunReloadTime <= 0.0f && gunShotTime <= 0.1f && remainingBulletsVal > 0 && magBulletsVal < magSize && !isDead )
        {
            animator.SetBool("reload", true);
            gunReloadTime = 2.5f;
            Invoke("reloaded", 2.0f);
        }
        else
        {
            animator.SetBool("reload", false);
        }
        updateText();
       
    }


    public void Being_shot(float damage) // getting hit from enemy
    {
        if (isDead) return;

        health -= damage;
        UpdateHealthDisplay();

        // print("player being shot, current health:" + health);

        if (health <= 0) 
        {
            Die();
        }
    }

    void Die() 
    {
        isDead = true;
        animator.SetBool("dead", true);
        characterMovement.isDead = true; // Set isDead in CharacterMovement
        StartCoroutine(RestartGame());
    }

    IEnumerator RestartGame() 
    {
        yield return new WaitForSeconds(10);
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    void UpdateHealthDisplay() 
    {
        if (healthDisplay != null) {
            healthDisplay.text = "Health: " + health;
        }
    }

    public void ReloadEvent(int eventNumber) // appearing and disappearing the handMag and gunMag
    {
        if(eventNumber==1)
        {
            handMag.GetComponent<SkinnedMeshRenderer>().enabled=true;
            gunMag.GetComponent<SkinnedMeshRenderer>().enabled=false;
        }
        if(eventNumber==2)
        {
            handMag.GetComponent<SkinnedMeshRenderer>().enabled=false;
            gunMag.GetComponent<SkinnedMeshRenderer>().enabled=true;
        }
    }

    void reloaded()
    {
        int newMagBulletsVal = Mathf.Min(remainingBulletsVal + magBulletsVal, magSize);
        int addedBullets = newMagBulletsVal - magBulletsVal;
        magBulletsVal = newMagBulletsVal;
        remainingBulletsVal = Mathf.Max(0, remainingBulletsVal - addedBullets);
        animator.SetBool("reloadAfterFire", false);
    }

    void updateText()
    {
        magBullets.text = magBulletsVal.ToString() ;
        remainingBullets.text = remainingBulletsVal.ToString();
    }

    void shotDetection() // Detecting the object which player shot 
    {
        ShootAtEnemy();

        RaycastHit rayHit;
        // we want everything except layer 8.
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

    void addEffects() // Adding muzzle flash, shoot sound and bullet hole on the wall
    {

    }

    void ShootAtEnemy() {

        // Debug.Log("shoot the enemy");

        lastShotTime = Time.time;

        Vector3 shootingDirection = (Enemy.transform.position - end.transform.position).normalized;
        shootingDirection = Quaternion.Euler(Random.Range(-accuracyRange, accuracyRange), Random.Range(-accuracyRange, accuracyRange), 0) * shootingDirection;

        RaycastHit rayHit;
        if (Physics.Raycast(end.transform.position, shootingDirection, out rayHit, shootingRange)) {
            if (rayHit.collider.gameObject == Enemy) {
                enemyGun enemyGunScript = enemygun.GetComponent<enemyGun>();
                if (enemyGunScript != null) {
                    enemyGunScript.Being_shot(bulletDamage);
                }
            }
        }
    }

    public void AddAmmo(int amount) {
        remainingBulletsVal += amount;
        updateText();  // Update the ammo display
    }

}
