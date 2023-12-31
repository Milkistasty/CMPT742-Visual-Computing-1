using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class EscapeDoor : MonoBehaviour {
    public Text winText;
    
    private void OnTriggerEnter(Collider other) {
        if (other.CompareTag("Player")) {
            // Player has reached the escape door
            WinGame();
        }
    }

    void WinGame() {
        winText.gameObject.SetActive(true);
        StartCoroutine(RestartGame());
    }

    IEnumerator RestartGame() {
        yield return new WaitForSeconds(10);
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
}
