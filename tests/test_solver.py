# tests/test_solver.py
import pytest
import numpy as np
import coin_problem_rs


def test_small_case():
    # 10円, 50円, 100円で 100円を作る。最大10枚。
    # 100: 1枚
    # 50*2: 2枚
    # 50*1 + 10*5: 6枚
    # 10*10: 10枚
    # 合計4通り
    target = 100
    max_coins = 10
    coins = np.array([10, 50, 100], dtype=np.int64)

    result = coin_problem_rs.solve(target, max_coins, coins)
    assert result == 4


def test_q05_known_result():
    # 以前の対話Q05の結果検証
    target = 1000
    max_coins = 15
    coins = np.array([10, 50, 100, 500], dtype=np.int64)

    result = coin_problem_rs.solve(target, max_coins, coins)
    assert result == 20


def test_zero_target():
    # 0円を作る方法は「何も出さない」の1通り
    coins = np.array([10], dtype=np.int64)
    assert coin_problem_rs.solve(0, 10, coins) == 1


def test_large_random():
    # クラッシュしないことの確認
    target = 10000
    max_coins = 100
    coins = np.array([10, 50, 100, 500], dtype=np.int64)
    result = coin_problem_rs.solve(target, max_coins, coins)
    assert result > 0
