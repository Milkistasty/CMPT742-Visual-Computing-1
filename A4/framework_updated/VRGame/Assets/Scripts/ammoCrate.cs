using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public class ammoCrate : MonoBehaviour {
    public int ammoAmount = 90;  // Amount of ammo to refill

    private void OnTriggerEnter(Collider other) {
        if (other.CompareTag("Player")) {
            Gun playerGun = other.GetComponent<Gun>();
            if (playerGun != null) {
                playerGun.AddAmmo(ammoAmount);
            }
        }
    }
}
