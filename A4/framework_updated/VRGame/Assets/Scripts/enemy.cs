using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class enemy : MonoBehaviour
{
    public Animator animator;
    public GameObject[] targets;
    public GameObject player;

    public bool isDead = false;

    public float moveSpeed = 5.0f;
    public float detectionRange = 30.0f;
    public float chaseDistance = 10.0f;
    private int currentTargetIndex = 0;
    private bool isChasingPlayer = false;

    // Start is called before the first frame update
    void Start()
    {
        // animator.SetTrigger("die");
    }

    // Update is called once per frame
    void Update()
    {
        if (!isDead)
        {
            if (detect_player())
            {
                animator.SetBool("player_detect", true);
                isChasingPlayer = true;
            }
            // else if (isChasingPlayer && !detect_player())
            // {
            //     isChasingPlayer = false;
            //     animator.SetBool("player_detect", false);
            // }

            if (isChasingPlayer)
            {
                MoveTowardsPlayer();
            }
            else
            {
                FollowPath();
            }
        }
    }

    bool detect_player()
    {
        Vector3 player_direction = player.transform.position - transform.position;
        float dist_player = Vector3.Distance(transform.position, player.transform.position);
        float angle_with_player = Vector3.Angle(player_direction, transform.forward);
        return (dist_player < detectionRange && Mathf.Abs(angle_with_player) < 30) || animator.GetBool("player_detect") ;
    }

    void MoveTowardsPlayer()
    {
        float distanceToPlayer = Vector3.Distance(transform.position, player.transform.position);
        if (distanceToPlayer > chaseDistance)
        {
            animator.SetBool("dist_at_10", false);
            MoveTowards(player.transform.position);
        }
        else if (distanceToPlayer <= chaseDistance)
        {
            animator.SetBool("dist_at_10", true);
        }
    }

    void FollowPath()
    {
        GameObject currentTarget = targets[currentTargetIndex];
        if (Vector3.Distance(transform.position, currentTarget.transform.position) < 1.0f)
        {
            currentTargetIndex = (currentTargetIndex + 1) % targets.Length;
        }
        MoveTowards(currentTarget.transform.position);
    }

    void MoveTowards(Vector3 target)
    {
        Vector3 directionToTarget = (target - transform.position).normalized;
        transform.LookAt(target);
        transform.position += directionToTarget * moveSpeed * Time.deltaTime;
    }
}
