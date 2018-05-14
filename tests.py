def check_types():
    from mypy import api
    res = api.run(['rl'])
    print('\n'.join(res[:-1]))

def test_StationaryKArmed():
    from rl.world.bandits import StationaryKArmed
    bandit = StationaryKArmed(3)
    assert bandit.k == 3
    
    sample_means = tuple(
        sum(bandit.pull_lever(i) for _ in range(1000)) /1000
            for i in range(bandit.k))
    assert all(abs(bandit.means[j]-sample_means[j])<0.5
        for j in range(bandit.k))

def test_EpsilonKArmedLearner():
    from rl.algos.bandit import EpsilonKArmedLearner
    from rl.world.bandits import StationaryKArmed
    bandit = StationaryKArmed(3)
    learner = EpsilonKArmedLearner(
        bandit=bandit,
        epsilon=0.1,
        alpha=lambda n: 1/n)
    assert learner.epsilon == 0.1
    print(learner.alpha(3))

if __name__ == '__main__':
    print('Checking types')
    check_types()
    print('Running StationaryKArmed tests')
    test_StationaryKArmed()
    print('Running bandit solver tests')
    test_EpsilonKArmedLearner()
    print('OK')