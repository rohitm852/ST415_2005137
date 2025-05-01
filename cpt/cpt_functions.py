# Class of CPT Functions
class CPTFunctions:

    def __init__(self, alpha_value, lambda_value, delta_value):
        # Initialise preference parameters
        self.alpha_value = alpha_value
        self.lambda_value = lambda_value
        self.delta_value = delta_value

    # This function computes the 'utility' of an outcome
    def value_function(self, winnings):
        # Inputs: winnings (integer), self.alpha_value (integer), self.lambda_value (integer)
        # Output: lottery_value (integer)

        # Output Errors if parameters are set to be out of their domains
        if self.alpha_value < 0 or self.alpha_value > 1:
            raise ValueError("The alpha value is out of bounds. It must be between 0 and 1.")
        if self.lambda_value < 1:
            raise ValueError("The lambda value is out of bounds. It must be greater than 1.")

        # Piecewise Kahneman and Tversky Value Function
        if winnings >= 0:
            lottery_value = winnings ** self.alpha_value
        else:
            lottery_value = (-self.lambda_value) * ((-winnings) ** self.alpha_value)

        return lottery_value

    # This function computes the weighting of an individual probability
    def probability_weighting_function(self,probability):
        # Inputs: probability (integer), self.delta (integer)
        # Output: weighted_probability (integer)

        # Output Error if parameter is set to be out of its domain
        if self.delta_value < 0 or self.delta_value > 1:
            raise ValueError("The delta value is out of bounds. It must be between 0 and 1.")

        # Kahneman and Tversky Probability Weighting Function
        numerator = (probability ** self.delta_value)
        denominator = (((probability ** self.delta_value) + ((1 - probability) ** self.delta_value)) ** (1 / self.delta_value))
        weighted_probability = numerator / denominator

        return weighted_probability

    def pi_values_function(self,outcomes, probabilities):
        # Inputs: outcomes (array), probabilities (array)
        # Output: sorted_outcomes (array), pi_values (array)

        # Combine and sort outcomes & probs in increasing order
        combined = sorted(zip(outcomes, probabilities))
        sorted_outcomes, sorted_probabilities = zip(*combined)
        sorted_outcomes = list(sorted_outcomes)
        sorted_probabilities = list(sorted_probabilities)

        # Create index list for negative to positive mapping (-m to n)
        index_list = []
        negatives = [x for x in sorted_outcomes if x < 0]
        positives = [x for x in sorted_outcomes if x > 0]
        zero = [x for x in sorted_outcomes if x == 0]
        index_list.extend(range(-len(negatives), 0))
        if zero:
            index_list.append(0)
        index_list.extend(range(1, len(positives) + 1))

        # Compute pi-values
        pi_values = []
        n = len(index_list)

        for i in range(n):
            idx = index_list[i]

            if idx >= 0:  # Positive/zero outcomes
                cumulative_p = sum(sorted_probabilities[i:n])
                remaining_p = sum(sorted_probabilities[i + 1:n])
            else:  # Negative outcomes
                cumulative_p = sum(sorted_probabilities[0:i + 1])
                remaining_p = sum(sorted_probabilities[0:i])

            w_cumulative = self.probability_weighting_function(cumulative_p)
            w_remaining = self.probability_weighting_function(remaining_p)
            pi_i = w_cumulative - w_remaining

            pi_values.append(pi_i)

        return sorted_outcomes, pi_values