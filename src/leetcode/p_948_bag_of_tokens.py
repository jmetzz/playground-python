"""https://leetcode.com/problems/bag-of-tokens/description

948. Bag of Tokens
Medium

You start with an initial power of power, an initial score of 0, and a bag of tokens
given as an integer array tokens, where each tokens[i] donates the value of token_i.

Your goal is to maximize the total score by strategically playing these tokens.
In one move, you can play an unplayed token in one of the two ways (but not both for the same token):

Face-up: If your current power is at least tokens[i], you may play token_i,
losing tokens[i] power and gaining 1 score.

Face-down: If your current score is at least 1, you may play token_i,
gaining tokens[i] power and losing 1 score.

Return the maximum possible score you can achieve after playing any number of tokens.


Example 1:
Input: tokens = [100], power = 50
Output: 0
Explanation: Since your score is 0 initially, you cannot play the token face-down.
You also cannot play it face-up since your power (50) is less than tokens[0] (100).

Example 2:
Input: tokens = [200,100], power = 150
Output: 1
Explanation: Play token1 (100) face-up, reducing your power to 50 and increasing your score to 1.
There is no need to play token0, since you cannot play it face-up to add to your score.
The maximum score achievable is 1.

Example 3:
Input: tokens = [100,200,300,400], power = 200
Output: 2
Explanation: Play the tokens in this order to get a score of 2:
Play token0 (100) face-up, reducing power to 100 and increasing score to 1.
Play token3 (400) face-down, increasing power to 500 and reducing score to 0.
Play token1 (200) face-up, reducing power to 300 and increasing score to 1.
Play token2 (300) face-up, reducing power to 0 and increasing score to 2.
The maximum score achievable is 2.


Constraints:
0 <= tokens.length <= 1000
0 <= tokens[i], power < 104
"""


def bag_of_tokens_score(tokens: list[int], power: int) -> int:
    tokens.sort()
    score, low, high = 0, 0, len(tokens) - 1
    while low <= high:
        if power >= tokens[low]:
            # face-up: increase score, but cost power
            # choose the lowest power tokens
            score += 1
            power -= tokens[low]
            low += 1
        elif score > 0 and low < high:
            # face_down: gain power, but decrease score
            # choose the highest power tokens
            # Note: the exception here is when there is only one token remaining.
            #       Since we lose a score point every time we play a token face-up,
            #       it would not maximize our score to play a token face-down
            #       unless there is one to play after it.
            power += tokens[high]
            score -= 1
            high -= 1
        else:
            # there is not enough score, power, or tokens to play
            # face-up or face-down, thus, return the score we have
            return score
    # all tokens were played, return the score we collected
    return score


if __name__ == "__main__":
    print(bag_of_tokens_score([100], 50))  # 0
    print(bag_of_tokens_score([200, 100], 150))  # 1
    print(bag_of_tokens_score([100, 200, 300, 400], 200))  # 2
