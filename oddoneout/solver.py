from nltk.corpus import wordnet as wn
from taxonomy import lowest_common_ancestor


class SimilarityScore:

    def __call__(self, word, other_words):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def is_recognized(self, word):
        raise NotImplementedError('Cannot call this method on abstract class.')


class TaxonomySimilarity(SimilarityScore):

    def __init__(self, taxonomy):
        super().__init__()
        self.taxonomy = taxonomy

    def __call__(self, word, other_words):
        spec, reason = lowest_common_ancestor(self.taxonomy,
                                              list(other_words),
                                              word)
        return 1.0/spec, reason

    def is_recognized(self, word):
        return self.taxonomy.is_instance(word)


def is_solvable(puzzle, similarity):
    options = puzzle.wordset + [puzzle.oddone]
    for opt in options:
        if not similarity.is_recognized(opt):
            return False
    return True


def rank_puzzle_choices(puzzle, sim):
    def score_choice(choice):
        others = set(choices) - {choice}
        return sim(choice, others)
    if not is_solvable(puzzle, sim):
        return None
    else:
        choices = puzzle.wordset + [puzzle.oddone]
        scores = [score_choice(choice) for choice in choices]
        result = sorted([(score, reason, choice)
                         for ((score, reason), choice) in zip(scores, choices)])
        return list(reversed(result))


def solve_puzzle(puzzle, similarity):
    ranks = rank_puzzle_choices(puzzle, similarity)
    if ranks is None or ranks[0][0] == ranks[1][0]:
        return None
    else:
        return ranks[0]


def verbose_logger(s):
    print(s)


def silent_logger(s):
    pass


def solve_puzzles(puzzles, model, logger=verbose_logger):
    correct = 0
    incorrect = 0
    unattempted = 0
    for puzzle in puzzles:
        solution = solve_puzzle(puzzle, model)
        if solution is None:
            logger('*ABSTAIN*: {}'.format(puzzle))
            unattempted += 1
        else:
            (score, reason, hypothesis) = solution
            if hypothesis == puzzle.oddone:
                logger('*CORRECT ({})* {}: {}'.format(reason,hypothesis, puzzle))
                correct += 1
            else:
                logger('*INCORRECT ({})* {}: {}'.format(reason,hypothesis, puzzle))
                incorrect += 1
                logger(' '.join(puzzle.wordset + [puzzle.oddone]))
                logger("Incorrect: " + str(hypothesis) + " should be " + str(puzzle.oddone))
    return correct, incorrect, unattempted


