# coding=utf-8
import os
import time

import numpy as np
from scipy.special import psi

from modules.distribution.categorical import Categorical
from modules.lda.plot import Plot
import collections
from sklearn.metrics.cluster import adjusted_rand_score


class LDA(object):
    """
    LDA (ギブスサンプリング)

    MLP(機械学習プロフェッショナルシリーズ)トピックモデル に準拠
    詳細は当該参考書 P71 参照

    各パラメータの詳細:

        alpha(np.ndarray):  トピックの出現頻度の偏りを表すパラメータ
        beta(np.ndarray):   語彙の出現頻度の偏りを表すパラメータ
        D(int):             文書数
        V(int):             全文書の中で現れる単語の種類数(≠単語数)
        K(int):             トピック数
        W(np.ndarray):      文書集合
        w_d(np.ndarray):    文書d
        w_dn(np.ndarray):   文書dのn番目の単語
        z_dn(np.ndarray):   文書dのn番目の単語のトピック
        N_d(np.ndarray):    文書dに含まれる単語数
        N_dk(np.ndarray):   文書dでトピックkが割り当てられた単語数
        θ_d(np.ndarray):    文章dでトピックkが割り当てられる確率
        N_k(np.ndarray):    文書集合全体でトピックkが割り当てられた単語数
        N_kv(np.ndarray):   文書集合全体で語彙vにトピックkが割り当てられた単語数
        φ_kv(np.ndarray):   トピックkのとき語彙vが生成される確率

    """

    def __init__(self, alpha, beta, data, label, D, V, K, N_d, save_path):
        """
        コンストラクター

        Args:
            alpha(np.ndarray):          トピックの出現頻度の偏りを表すパラメータ
            beta(np.ndarray):           語彙の出現頻度の偏りを表すパラメータ
            data(list[np.ndarray]):     処理するデータ (Bag-of-Words形式)
            label(np.ndarray):          トピックの正解ラベル(ARI用)
            N_d(np.ndarray):            それぞれの文書の単語数
            D(int):                     文書数
            V(np.ndarray):              モダリティごとに全文書の中で現れる単語の種類数(≠単語数)
            K(int):                     トピック数

        """
        self.alpha = alpha
        self.beta = beta

        self.label = label

        self.D = D
        self.V = V
        self.K = K
        self.N_d = N_d
        self.w_dn = [np.zeros(shape=self.N_d[d], dtype=int) for d in range(self.D)]
        self.z_dn = [np.zeros(shape=self.N_d[d], dtype=int) - 1 for d in range(self.D)]
        self.N_dk = np.zeros(shape=(self.D, self.K), dtype=int)
        self.N_k = np.zeros(shape=self.K, dtype=int)
        self.N_kv = np.zeros(shape=(self.K, self.V), dtype=int)
        self.theta = np.zeros(shape=(self.D, self.K))
        self.topic = np.zeros(shape=self.D, dtype=int)
        self.ari = []
        self.w_dn = self.bag_of_words_to_sentence(data=data, D=self.D, V=self.V, N_d=self.N_d)
        self.save_path = save_path
        self.train_mode = True
        print("ドキュメント数: {}, 単語数: {}, 語彙数: {}, トピック数: {}".format(self.D, self.N_d.sum() / self.N_d.size, self.V, self.K))

    def gibbs_sampling(self, iteration, interval):
        """
        ギブスサンプリング(メイン処理)

        Args:
            iteration(int): 試行回数
            interval(int):  データを保存する間隔

        """
        epoch = iteration / interval

        for i in range(epoch):
            start = time.time()
            for b in range(interval):
                for d in range(self.D):
                    for n in range(self.N_d[d]):
                        # w_dn: ドキュメント(d番目)の単語(n番目)の語彙
                        # z_dn: ドキュメント(d番目)の単語(n番目)のトピック
                        w_dn = self.w_dn[d][n]
                        z_dn = self.z_dn[d][n]

                        # トピックが割り振られているなら以下の処理を行う
                        # 1. ドキュメント(d番目)内における、トピック(z_dn)の出現数のカウントを1減らす
                        # 2. ドキュメント全体で、単語(w_dn)の内、トピック(z_dn)が割り当てられた単語数のカウントを1減らす
                        # 3. ドキュメント全体で、トピック(z_dn)が割り当てられた単語数のカウントを1減らす
                        if z_dn >= 0:
                            self.N_dk[d][z_dn] -= 1
                            self.N_kv[z_dn][w_dn] -= 1
                            self.N_k[z_dn] -= 1

                        # サンプリング確率を計算
                        self.theta[d] = self.calc_topic_probability(
                            alpha=self.alpha, beta=self.beta, N_dk=self.N_dk, N_kv=self.N_kv, N_k=self.N_k, w_dn=w_dn, d=d
                        )

                        # トピックをサンプリング(トピックの更新)
                        updated_z_dn = self.sampling_topic_from_categorical(self.theta[d])

                        # 更新したトピックで以下の処理を行う
                        # 1. ドキュメント(d番目)内における、トピック(z_dn)の出現数のカウントを1増やす
                        # 2. ドキュメント全体で、単語(w_dn)の内、トピック(z_dn)が割り当てられた単語数のカウントを1増やす
                        # 3. ドキュメント全体で、トピック(z_dn)が割り当てられた単語数のカウントを1増やす
                        self.N_dk[d][updated_z_dn] += 1
                        self.N_kv[updated_z_dn][w_dn] += 1
                        self.N_k[updated_z_dn] += 1

                        # 更新したトピックを反映
                        self.z_dn[d][n] = updated_z_dn

                # ハイパーパラメータを更新(学習モード時のみ)
                # if self.train_mode:
                #     self.alpha = self.update_alpha(alpha=self.alpha, D=self.D, N_dk=self.N_dk, N_d=self.N_d)
                #     self.beta = self.update_beta(beta=self.beta, K=self.K, N_kv=self.N_kv, N_k=self.N_k)

            elapsed_time = time.time() - start
            self.topic = np.argmax(self.theta, axis=1)
            print(self.topic.shape, self.label.shape)
            ari = self.calc_adjusted_rand_score(data=self.topic, label=self.label)
            print("\nIteration: {}, Time: {:.2f}s({:.2f}s/iter)".format((i + 1) * interval, elapsed_time, elapsed_time / interval))
            print("ARI = {}".format(ari))
            self.ari.append(ari)
            self.save_result()
            Plot.plot_ari(self.save_path, i, epoch, self.ari)

        # Plot().plot_theta(self.D, self.K, self.theta)

    @staticmethod
    def calc_topic_probability(alpha, beta, N_dk, N_kv, N_k, w_dn, d):
        """
        トピックのサンプリング確率
        P(z_dn = k | W, Z/dn, α, β)を求める

        Args:
            alpha(np.ndarray):  トピックの出現頻度の偏りを表すパラメータ
            beta(np.ndarray):   語彙の出現頻度の偏りを表すパラメータ
            w_dn(int):          文書dのn番目の単語
            N_dk(np.ndarray):   文書dでトピックkが割り当てられた単語数
            N_k(np.ndarray):    文書集合全体でトピックkが割り当てられた単語数
            N_kv(np.ndarray):   文書集合全体で語彙vにトピックkが割り当てられた単語数
            d(int):             d番目の文書

        Returns:
            np.ndarray:         d番目の文書における、それぞれのトピックの確率

        """
        N_dk_dn = N_dk[d]
        N_kw_dn_dn = N_kv[:, w_dn]
        N_k_dn = N_k
        a = N_dk_dn + alpha
        b = N_kw_dn_dn + beta[w_dn]
        c = N_k_dn + beta.sum()
        p = a * (b / c)
        vector = np.array(p)

        return vector / vector.sum()

    def update_alpha(self, alpha, D, N_dk, N_d):
        """
        ハイパーパラメータ α を更新

        Args:
            alpha(np.ndarray):  トピックの出現頻度の偏りを表すパラメータ
            D(int):             文書数
            N_d(np.ndarray):    文書dに含まれる単語数
            N_dk(np.ndarray):   文書dでトピックkが割り当てられた単語数

        Returns:
            np.ndarray:         更新したハイパーパラメータ α

        """
        # ディガンマ関数に0をいれるとマイナス無限大に発散しWarningが出るが、式全体としては問題無いため警告を非表示にする

        with np.errstate(invalid='ignore'):
            a = np.array([self.digamma(N_dk[d] + alpha) for d in range(D)]).sum(axis=0) - D * (self.digamma(alpha))
            b = self.digamma(N_d + alpha.sum()).sum() - D * self.digamma(alpha.sum())

            # ライブラリの仕様上 Nan, Infinite, マイナス値 が発生する場合がある。これらは本来、値が0なので0に置き換える。
            # 原因は以下の2種類
            # 1. ディガンマ関数は0でマイナス無限大に発散するため、非常に小さい値を入れると Nan や Infinite が発生する
            # 2. N_dk[d]が0のとき計算式として0になるが、場合によって計算結果の誤差によってマイナス値が発生する (α は必ず正の実数値)
            a[(np.isnan(a)) | (np.isinf(a)) | (a < 0)] = 0

        return alpha * (a / b)

    def update_beta(self, beta, K, N_kv, N_k):
        """
        ハイパーパラメータ β を更新

        Args:
            beta(np.ndarray):
            K(int):             トピック数
            N_k(np.ndarray):    文書集合全体でトピックkが割り当てられた単語数
            N_kv(np.ndarray):   文書集合全体で語彙vにトピックkが割り当てられた単語数

        Returns:
            np.ndarray:         更新したハイパーパラメータ β

        """
        # ディガンマ関数に0をいれるとマイナス無限大に発散しWarningが出るが、式全体としては問題無いため警告を非表示にする
        with np.errstate(invalid='ignore'):
            a = np.array([self.digamma(N_kv[k] + beta) for k in range(K)]).sum(axis=0) - K * (self.digamma(beta))
            b = self.digamma(N_k + beta.sum()).sum() - K * self.digamma(beta.sum())

            # ライブラリの仕様上 Nan, Infinite, マイナス値 が発生する場合がある。これらは本来、値が0なので0に置き換える。
            # 原因は以下の2種類
            # 1. ディガンマ関数は0でマイナス無限大に発散するため、非常に小さい値を入れると Nan や Infinite が発生する
            # 2. N_kv[k]が0のとき計算式として0になるが、場合によって計算結果の誤差によってマイナス値が発生する (β は必ず正の実数値)
            a[(np.isnan(a)) | (np.isinf(a)) | (a < 0)] = 0

        return beta * (a / b)

    def save_result(self):
        """
        各種パラメータを保存

        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        alpha_path = os.path.join(self.save_path, "alpha.txt")
        beta_path = os.path.join(self.save_path, "beta.txt")
        theta_path = os.path.join(self.save_path, "theta.txt")
        z_dn_path = os.path.join(self.save_path, "z_dn.txt")
        N_dk_path = os.path.join(self.save_path, "N_dk.txt")
        N_k_path = os.path.join(self.save_path, "N_k.txt")
        N_kv_path = os.path.join(self.save_path, "N_kv.txt")
        z_dn_5_path = os.path.join(self.save_path, "z_dn_5.txt")
        topic_path = os.path.join(self.save_path, "topic.txt")
        ari_path = os.path.join(self.save_path, "ARI.txt")

        array = np.zeros(shape=(self.D, 5), dtype=int) - 1
        topics = set()
        for d in range(self.D):
            z_dn = self.z_dn[d]
            c = collections.Counter(z_dn).most_common(5)
            topics = topics.union(set(z_dn))
            for i in range(len(c)):
                array[d][i] = c[i][0]
        topics = np.array(sorted(list(topics)), dtype=int)

        np.savetxt(z_dn_5_path, array, fmt="%d")
        np.savetxt(topic_path, topics, fmt="%d")
        np.savetxt(alpha_path, self.alpha)
        np.savetxt(beta_path, self.beta)
        np.savetxt(theta_path, self.theta)
        np.savetxt(z_dn_path, self.z_dn, fmt="%d")
        np.savetxt(N_dk_path, self.N_dk, fmt="%d")
        np.savetxt(N_k_path, self.N_k, fmt="%d")
        np.savetxt(N_kv_path, self.N_kv, fmt="%d")
        np.savetxt(ari_path, self.ari)

    def change_train_mode(self):
        self.train_mode = True

    def change_eval_mode(self):
        self.train_mode = False

    @staticmethod
    def bag_of_words_to_sentence(data, D, V, N_d):
        """
        Bag-of-words形式のデータを、文書形式に変換する

        Args:
            data(list[np.ndarray]):   処理するデータ (Bag-of-Words形式)
            N_d(np.ndarray):    それぞれの文書の単語数
            D(int):             文書数
            V(int):             全文書の中で現れる単語の種類数(≠単語数)

        """
        x = [np.zeros(shape=N_d[d], dtype=int) for d in range(D)]
        for d in range(D):
            x[d] = np.array([v for v in range(V) for i in range(data[d][v])])

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

    @staticmethod
    def digamma(z):
        """
        ディガンマ関数(Ψ関数)(ガンマ関数の対数微分)の計算
        ディガンマ関数: Γ'(z)/Γ(z)

        Args:
            z(np.ndarray): 入力

        Returns:
            np.ndarray: 対数微分値

        """
        return psi(z)
