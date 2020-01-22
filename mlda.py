# coding=utf-8
import os
import time

import numpy as np
from scipy.special import psi

from modules.distribution.categorical import Categorical
from modules.lda.plot import Plot
import collections
from sklearn.metrics.cluster import adjusted_rand_score


class MLDA(object):
    """
    MLDA (ギブスサンプリング)

    各パラメータの詳細:

        alpha(np.ndarray):                  トピックの出現頻度の偏りを表すパラメータ
        beta(np.ndarray):                   語彙の出現頻度の偏りを表すパラメータ
        D(int):                             データ数
        K(int):                             トピック数
        M(int):                             モダリティ数
        V_m(np.ndarray):                    モダリティmのデータ全体で現れる単語の種類数(≠単語数)
        w_d(np.ndarray):                    データd
        w_dmn(list[list[np.ndarray]]):      データdのモダリティmのn番目の単語
        z_dmn(np.ndarray):                  データdのモダリティmのn番目の単語のトピック
        N_dm(list[np.ndarray]):             データdのモダリティmに含まれる単語数
        N_dmk(list[np.ndarray]):            データdのモダリティmでトピックkが割り当てられた単語数
        θ_d(np.ndarray):                    文章dでトピックkが割り当てられる確率
        N_mk(np.ndarray):                   モダリティmのデータ全体でトピックkが割り当てられた単語数
        N_kmv(list[list[np.ndarray]]):      モダリティmのデータ全体で語彙vにトピックkが割り当てられた単語数
        φ_kv(np.ndarray):                   トピックkのとき語彙vが生成される確率

    """

    def __init__(self, alpha, beta, data, label, D, V_m, K, N_dm, M, save_path):
        """
        コンストラクター

        Args:
            alpha(np.ndarray):                  トピックの出現頻度の偏りを表すパラメータ
            beta(list[np.ndarray]):             語彙の出現頻度の偏りを表すパラメータ
            data(list[list[np.ndarray]]):       処理するデータ (Bag-of-Words形式)
            label(np.ndarray):                  トピックの正解ラベル(ARI用)
            D(int):                             データ数
            K(int):                             トピック数
            M(int):                             モダリティ数
            V_m(np.ndarray):                    モダリティmのデータ全体で現れる単語の種類数(≠単語数)
            N_dm(list[np.ndarray]):             それぞれのデータのモダリティごとの単語数
            save_path(str):                     結果の保存用のパス

        """
        self.label = label
        self.save_path = save_path

        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.K = K
        self.M = M
        self.V_m = V_m
        self.N_dm = N_dm
        self.w_dmn = self.bag_of_words_to_sentence(data=data, D=self.D, M=self.M, V_m=self.V_m, N_dm=self.N_dm)
        self.z_dmn = [[(np.zeros(shape=self.N_dm[d][m], dtype=int) - 1) for m in xrange(self.M)] for d in xrange(self.D)]
        self.N_dmk = [np.zeros(shape=(self.M, self.K), dtype=int) for d in xrange(self.D)]
        self.N_mk = np.zeros(shape=(self.M, self.K), dtype=int)
        self.N_kmv = [[np.zeros(shape=self.V_m[m], dtype=int) for m in xrange(self.M)] for k in xrange(self.K)]
        self.theta = np.zeros(shape=(self.D, self.K), dtype=float)
        self.topic = np.zeros(shape=self.D, dtype=int)
        self.ari = []

        print("ドキュメント数: {}, モダリティ数: {}".format(self.D, self.M))
        print("単語数: {}, 語彙数: {}, トピック数: {}".format(self.N_dm[0], self.V_m, self.K))

    def gibbs_sampling(self, iteration, interval=1):
        """
        ギブスサンプリング(メイン処理)

        Args:
            iteration(int): 試行回数
            interval(int):  途中結果の表示とデータの保存を行う間隔

        """
        epoch = iteration / interval

        total_start = time.time()
        for i1 in xrange(epoch):
            start = time.time()
            for i2 in xrange(interval):
                for d in xrange(self.D):
                    for m in xrange(self.M):
                        for n in xrange(self.N_dm[d][m]):
                            # w_dmn: ドキュメント(d番目)のモダリティ(m番目)の単語(n番目)の語彙
                            # z_dmn: ドキュメント(d番目)のモダリティ(m番目)の単語(n番目)のトピック
                            w_dmn = self.w_dmn[d][m][n]
                            z_dmn = self.z_dmn[d][m][n]

                            # トピックが割り振られているなら以下の処理を行う
                            # 1. ドキュメント(d番目)内における、トピック(z_dn)の出現数のカウントを1減らす
                            # 2. ドキュメント全体で、単語(w_dn)の内、トピック(z_dn)が割り当てられた単語数のカウントを1減らす
                            # 3. ドキュメント全体で、トピック(z_dn)が割り当てられた単語数のカウントを1減らす
                            if z_dmn >= 0:
                                self.N_dmk[d][m][z_dmn] -= 1
                                self.N_kmv[z_dmn][m][w_dmn] -= 1
                                self.N_mk[m][z_dmn] -= 1

                            # サンプリング確率を計算
                            theta = self.calc_topic_probability(
                                alpha=self.alpha, beta=self.beta, N_dmk=self.N_dmk, N_kmv=self.N_kmv, N_mk=self.N_mk, w_dmn=w_dmn, d=d, m=m, K=self.K
                            )

                            # トピックをサンプリング(トピックの更新)
                            updated_z_dmn = self.sampling_topic_from_categorical(theta)

                            # 更新したトピックで以下の処理を行う
                            # 1. ドキュメント(d番目)内における、トピック(z_dn)の出現数のカウントを1増やす
                            # 2. ドキュメント全体で、単語(w_dn)の内、トピック(z_dn)が割り当てられた単語数のカウントを1増やす
                            # 3. ドキュメント全体で、トピック(z_dn)が割り当てられた単語数のカウントを1増やす
                            self.N_dmk[d][m][updated_z_dmn] += 1
                            self.N_kmv[updated_z_dmn][m][w_dmn] += 1
                            self.N_mk[m][updated_z_dmn] += 1

                            # 更新したトピックを反映
                            self.z_dmn[d][m][n] = updated_z_dmn

            # データの保存と表示
            elapsed_time = time.time() - start
            z = np.zeros(shape=(self.D, self.K), dtype=int)
            for m in range(self.M):
                z += np.array([self.N_dmk[d][m] for d in range(self.D)])
            self.topic = np.argmax(z, axis=1)
            ari = self.calc_adjusted_rand_score(data=self.topic, label=self.label)
            print("\nIteration: {}, Time: {:.2f}s({:.2f}s/iter), Total Time: {:.2f}s".format((i1 + 1) * interval, elapsed_time, elapsed_time / interval,
                                                                                             time.time() - total_start))
            print("ARI = {}".format(ari))
            self.ari.append(ari)
            self.save_result()
            Plot.plot_ari(self.save_path, iteration, self.ari)

    @staticmethod
    def calc_topic_probability(alpha, beta, N_dmk, N_kmv, N_mk, w_dmn, d, m, K):
        """
        トピックのサンプリング確率
        P(z_dn = k | W, Z/dn, α, β)を求める

        Args:
            alpha(np.ndarray):              トピックの出現頻度の偏りを表すパラメータ
            beta(list(np.ndarray)):         語彙の出現頻度の偏りを表すパラメータ
            w_dmn(int):                     データdのn番目の単語
            N_dmk(list[np.ndarray]):        データdのモダリティmでトピックkが割り当てられた単語数
            N_kmv(list[list[np.ndarray]]):  モダリティmのデータ全体で語彙vにトピックkが割り当てられた単語数
        N_mk(np.ndarray):                   モダリティmのデータ全体でトピックkが割り当てられた単語数
            d(int):                         d番目のデータ
            m(int):                         d番目のデータのn番目の単語
            K(int):                         トピック数

        Returns:
            np.ndarray:                     データdのモダリティmの単語nにおけるそれぞれのトピックの生起確率

        """
        N_dk_dn = N_dmk[d].sum(axis=0)
        N_kw_dn_dn = np.array([N_kmv[k][m][w_dmn] for k in xrange(K)])
        N_k_dn = N_mk[m]
        a = N_dk_dn + alpha
        b = N_kw_dn_dn + beta[m][w_dmn]
        c = N_k_dn + beta[m].sum()
        p = a * (b / c)
        vector = np.array(p)
        return vector / vector.sum()

    def save_result(self):
        """
        各種パラメータを保存

        """
        # 保存するディレクトリを作成
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # モダリティごとの保存先を作成
        for m in range(self.M):
            modarity_path = os.path.join(self.save_path, str(m))
            if not os.path.exists(modarity_path):
                os.mkdir(modarity_path)

        alpha_path = os.path.join(self.save_path, "alpha.txt")
        beta_path = os.path.join(self.save_path, "beta.txt")
        topic_path = os.path.join(self.save_path, "topic.txt")
        ari_path = os.path.join(self.save_path, "ARI.txt")

        for m in range(self.M):
            modarity_path = os.path.join(self.save_path, str(m))
            z_dn_path = os.path.join(modarity_path, "z_dn.txt")
            N_dk_path = os.path.join(modarity_path, "N_dk.txt")
            N_k_path = os.path.join(modarity_path, "N_k.txt")
            N_kv_path = os.path.join(modarity_path, "N_kv.txt")
            np.savetxt(z_dn_path, [self.z_dmn[d][m] for d in range(self.D)], fmt="%d")
            np.savetxt(N_dk_path, [self.N_dmk[d][m] for d in range(self.D)], fmt="%d")
            np.savetxt(N_k_path, self.N_mk[m], fmt="%d")
            np.savetxt(N_kv_path, [self.N_kmv[k][m] for k in range(self.K)], fmt="%d")

        np.savetxt(topic_path, self.topic, fmt="%d")
        np.savetxt(alpha_path, self.alpha)
        np.savetxt(beta_path, self.beta)
        np.savetxt(ari_path, self.ari)

    @staticmethod
    def bag_of_words_to_sentence(data, D, M, V_m, N_dm):
        """
        Bag-of-words形式のデータを、データ形式に変換する

        Args:
            data(list[list[np.ndarray]]):       処理するデータ (Bag-of-Words形式)
            D(int):                             データ数
            M(int):                             モダリティ数
            V_m(np.ndarray):                    モダリティmのデータ全体で現れる単語の種類数(≠単語数)
            N_dm(list[np.ndarray]):             データdのモダリティmに含まれる単語数

        Returns:
            list[list[np.ndarray]]:             データdのモダリティmのn番目の単語(w_dmn)

        """
        x = [[np.zeros(shape=N_dm[d][m], dtype=int) - 1 for m in xrange(M)] for d in xrange(D)]
        for d in xrange(D):
            for m in xrange(M):
                x[d][m] = np.array([v for v in xrange(V_m[m]) for i in xrange(data[d][m][v])])

        return x

    @staticmethod
    def sampling_topic_from_categorical(pi):
        """
        カテゴリカル分布(パラメータπ)からトピックをサンプリング

        Args:
            pi:     カテゴリカル分布のパラメータπ

        Returns:
            int:    サンプリングされたトピックのID

        """
        vector = Categorical.sampling(pi=pi)
        return np.where(vector == 1)[0][0]

    @staticmethod
    def calc_adjusted_rand_score(data, label):
        """
        ARI(adjusted_rand_score)を計算する

        Args:
            data(np.ndarray):   分類結果
            label(np.ndarray):  正解ラベル

        Returns:
            float:  ARI値 (0.0 ~ 1.0)

        """
        return adjusted_rand_score(data, label)
