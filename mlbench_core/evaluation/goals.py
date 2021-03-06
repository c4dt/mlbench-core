def _add_detailed_times(result, tracker):
    compute_time = tracker.get_total_compute_time()

    if compute_time:
        result += ", Compute: {} seconds".format(compute_time)

    communication_time = tracker.get_total_communication_time()

    if communication_time:
        result += ", Communication: {} seconds".format(communication_time)

    return result


def time_to_accuracy_goal(threshold):
    def _time_to_accuracy_goal(metric_name, value, tracker):
        if metric_name != "val_global_Prec@1":
            return None
        if value >= threshold:
            duration = tracker.get_total_train_time()

            result = (
                "{0:02d}% Top 1 Validation Accuracy reached in {1:.3f} "
                "seconds".format(threshold, duration)
            )

            result = _add_detailed_times(result, tracker)

            return result

        return None

    return _time_to_accuracy_goal


def task1_time_to_accuracy_goal():
    """ Accuracy over Time target for benchmark task 1: Image classification

    Target is 80% accuracy

    Return:
        func: time_time_to_accuracy_goal with threshold = 80
    """
    return time_to_accuracy_goal(80)


def task1_time_to_accuracy_light_goal():
    """ Accuracy over Time target for benchmark task 1: Image classification
    (Light)

    Light target is 70% accuracy

    Return:
        func: time_time_to_accuracy_goal with threshold = 70
    """
    return time_to_accuracy_goal(70)


def task2_time_to_accuracy_goal():
    """Time to accuracy goal for benchmark task 2: Linear binary classifier

    Target is an accuracy of 89%

    Return:
        func: time_time_to_accuracy_goal with threshold = 89
    """
    return time_to_accuracy_goal(89)


def task2_time_to_accuracy_light_goal():
    """Time to perplexity goal for benchmark task 2: Linear binary classifier

    Target is an accuracy of 80%

    Return:
        func: time_time_to_accuracy_goal with threshold = 80
    """
    return time_to_accuracy_goal(80)


def task3_time_to_preplexity_goal(metric_name, value, tracker):
    """Time to perplexity goal for benchmark task 3: Language Modelling

    Target is a perplexity of 50

    Args:
        metric_name(str): Name of the metric to test the value for,
        only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object
        used for the current run
    Return:
        result (str) or `None` if target is not reached
    """

    if metric_name != "val_global_Perplexity":
        return None

    if value <= 50:
        duration = tracker.get_total_train_time()
        result = "Validation perplexity of 50 reached in {0:.3f} seconds".format(
            duration
        )

        result = _add_detailed_times(result, tracker)

        return result

    return None


def task3_time_to_preplexity_light_goal(metric_name, value, tracker):
    """Time to perplexity goal for benchmark task 3: Language Modelling

    Target is a perplexity of 50

    Args:
        metric_name(str): Name of the metric to test the value for,
        only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object
        used for the current run
    Return:
        result (str) or `None` if target is not reached
    """

    if metric_name != "val_global_Perplexity":
        return None

    if value <= 100:
        duration = tracker.get_total_train_time()
        result = "Validation perplexity of 50 reached in {0:.3f} seconds".format(
            duration
        )

        result = _add_detailed_times(result, tracker)

        return result

    return None


def task4_time_to_bleu_goal(threshold=24):
    """Time to BLEU-score goal for benchmark task 4: GNMT machine translation"""

    def _time_to_bleu_goal(metric_name, value, tracker):
        if metric_name != "val_global_BLEU-Score":
            return None

        if value >= threshold:
            duration = tracker.get_total_train_time()
            result = "Validation BLEU-Score of {0} reached in {1:.3f} seconds".format(
                threshold, duration
            )

            result = _add_detailed_times(result, tracker)

            return result

        return None

    return _time_to_bleu_goal
