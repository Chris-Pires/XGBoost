import numpy as np
import pandas as pd
from functools import reduce
import operator


dataframe = pd.DataFrame([['Chélonien', 'tortue', 0, 4, 0, 1],
                          ['actynoptérygien', 'thon', 0, 0, 0, 0],
                          ['actynoptérygien', 'brochet', 0, 0, 0, 0],
                          ['mammifère', 'chien', 0, 4, 1, 0],
                          ['mammifère', 'chat', 0, 4, 1, 0],
                          ['mammifère', 'éléphant', 0, 4, 1, 0],
                          ['mammifère', 'souris', 0, 4, 1, 0],
                          ['insecte', 'fourmi', 0, 6, 0, 0],
                          ['poisson', 'poisson-clown', 0, 0, 0, 0],
                          ['Chélonien', 'tortue marine', 0, 4, 0, 1],
                          ['mammifère', 'lion', 0, 4, 1, 0],
                          ['mammifère', 'girafe', 0, 4, 1, 0],
                          ['mammifère', 'singe', 0, 4, 1, 0],
                          ['crocodilien', 'aligator', 1, 4, 0, 1],
                          ['reptile', 'gecko', 0, 4, 0, 2],
                          ['reptile', 'lézard', 0, 4, 0, 2],
                          ['reptile', 'tortue terrestre', 0, 4, 0, 2],
                          ['oiseau', 'canard', 1, 2, 0, 0],
                          ['oiseau', 'perruche', 1, 2, 0, 0],
                          ['oiseau', 'corbeau', 1, 2, 0, 0],
                          ['reptile', 'iguane', 0, 4, 0, 0],
                          ['reptile', 'gecko', 0, 4, 0, 0],
                          ['reptile', 'serpent à sonnette', 0, 0, 0, 0],
                          ],
                          columns=['classification', 'espèce', 'gésier', 'nb membres', 'poil', 'carapace dorsal et ventral'])

dataframe_classif = dataframe.loc[:, ('classification', 'gésier', 'nb membres', 'poil', 'carapace dorsal et ventral')]
class node:

   def __init__(self, feature, value, cost, nature, left_branch, right_branch, depth, pop):
      self.feature = feature
      self.value = value
      self.nature = nature
      self.left_branch = left_branch
      self.right_branch = right_branch
      self.depth = depth
      self.pop = pop

   def __split__(self):
      if self.nature == 'quanti' :
          return self.feature + ' <= ' + str(self.value)
      elif self.nature == 'quali' :
          return self.feature + ' == ' + str(self.value)


class decision_tree:
    """ Cette classe a pour but de créer un algorithme d'apprentissage automatique
    d'arbres de décision classifieur"""

    def __init__(self, target, dataframe, max_depth):
        """Cette fonction a pour but d'initialiser les variables essentiel à la
        construction de notre algorithme.
       INPUT
        - target : la variable cible qu'il faudra classifier
        - dataframe : les données d'apprentissage
        - max_deapth : la profondeur maximal de l'arbre à entraîner
       """
        self.max_depth = max_depth
        self.target = target
        self.dataframe = dataframe

    def __quanti_split__(self, feature, value, dataset):
        """Cette fonction split un jeu de données en fonction
        de la valeur 'value' de la variable quantitative 'feature' passé en paramètre
        INPUT
         - feature : integer correspondant à la variable à séparer
         - value : integer correspond à la valeur à laquelle séparer notre jeu de données
         - dataset : pandas dataframe à séparer
        OUTPUT
         - left : dataframe avec les données où 'feature' est plus petit ou égale à 'value'
         - right : dataframe avec les données où 'feature' est plus grande que 'value'
        """

        left = dataset[dataset.loc[:, feature] <= value]
        right = dataset[dataset.loc[:, feature] > value]

        return left, right

    def __quali_split__(self, feature, value, dataset):
        """Cette fonction split un jeu de données en fonction
        de la valeur 'value' de la variable qualitative 'feature' passé en paramètre
        INPUT
         - feature : integer correspondant à la variable à séparer
         - value : integer correspond à la valeur à laquelle séparer notre jeu de données
         - dataset : pandas dataframe à séparer
        OUTPUT
         - left : dataframe avec les données où 'feature' est égale 'value'
         - right : dataframe avec les données où 'feature' est différent 'value'
        """

        left = dataset[dataset.loc[:, feature] == value]
        right = dataset[dataset.loc[:, feature] != value]

        return left, right

    def __add_dict__(self, prec_dict, new_dict):
        """ Cette fonction fusionne des dictionnaires
        en sommant les valeurs des clefs similaires
        INPUT
           - prec_dict : dictionnaire de fusion
           - new_dict : dictionnaire à fusionner
        OUTPUT
           - prec_dict : dictionnaire avec en clef la 'classe'
           et en 'valeur' son occurence dans le dataset"""
        if list(new_dict.keys())[0] in prec_dict:
            prec_dict[list(new_dict.keys())[0]] += 1
        else:
            prec_dict[list(new_dict.keys())[0]] = 1
        return prec_dict

    def __gini__(self, dataset):
        """Calculer les indices Gini du dataset passé en paramètre
        INPUT
           - dataset : dataframe contenant la variable 'self.target'
        OUTPUT
           - impurity : l'impureté calculée à partir du dataset
        """
        rows = dataset[self.target]
        class_dict = list(map(lambda x: {x: 1}, rows))
        class_dict_sum = reduce(self.__add_dict__, class_dict)
        occu_class = np.fromiter(reduce(self.__add_dict__,
                                        list(map(lambda x: {x: 1}, rows))).values(),
                                 dtype=float)
        pop = np.sum(occu_class)
        impurity = 1 - np.sum((occu_class / pop) ** 2)
        return impurity

    def __split_evaluator__(self, left_dataset, right_dataset):
        """ Calculer le coût d'une séparation d'un noeud en deux branches
        INPUT
           - left_dataset : dataset de la branche de gauche
           - right_dataset : dataset de la branche de droite
        OUTPUT
           - cost : coût de la séparation"""
        left_eval = self.__gini__(left_dataset)
        nb_left = left_dataset.shape[0]
        right_eval = self.__gini__(right_dataset)
        nb_right = right_dataset.shape[0]
        nb_tot = nb_left + nb_right
        cost = nb_left / nb_tot * left_eval + nb_right / nb_tot * right_eval
        return cost

    def __test_quali__(self, dataset, feature):
        """ Tester toutes les séparations possibles d'une variable qualitative
        INPUT
           - dataset : dataset à évaluer
           - feature : variable du dataset à évaluer
        OUTPUT
           - df_eval : dataframe contenant le coût de chaque séparation"""

        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        for value in dataset.loc[:, feature].unique():
            left, right = self.__quali_split__(feature, value, dataset)
            cost_result = self.__split_evaluator__(left, right)
            df_eval = df_eval.append(pd.DataFrame([[feature,
                                                    value,
                                                    'quali',
                                                    cost_result]],
                                                  columns=('feature', 'value', 'nature', 'cost')))
        return df_eval

    def __test_quanti__(self, dataset, feature):
        """ Tester toutes les séparations possibles d'une variable quantitative
        INPUT
           - dataset : dataset à évaluer
           - feature : variable du dataset à évaluer
        OUTPUT
           - df_eval : dataframe contenant le coût de chaque séparation"""

        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        value_to_test = (dataset.loc[:, feature].sort_values()[1:].values + dataset.loc[:, feature].sort_values()[
                                                                            :-1].values) / 2
        for value in value_to_test:
            left, right = self.__quanti_split__(feature, value, dataset)
            cost_result = self.__split_evaluator__(left, right)
            df_eval = df_eval.append(pd.DataFrame([[feature,
                                                    value,
                                                    'quanti',
                                                    cost_result]],
                                                  columns=('feature', 'value', 'nature', 'cost')))
        return df_eval

    def __find_best_split__(self, dataset):
        """ Trouver la meilleure séparation de notre jeu de données
        INPUT
           - dataset : jeu de données à séparer
        OUTPUT
           - def_eval : dataset contenant 'feature' variable à séparer, 'value'
           la valeur à laquelle séparer la variable, 'nature' la nature de la
           variable et 'cost' le coût de cette séparation"""
        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        columns = dataset.columns[np.logical_not(dataset.columns == self.target)]
        for column in columns:
            if len(dataset[column].unique()) >= 10:
                df_eval = df_eval.append(self.__test_quanti__(dataset, column))
            elif len(dataset[column].unique()) > 1:
                df_eval = df_eval.append(self.__test_quali__(dataset, column))

        df_eval = df_eval.reset_index(drop=True)

        idx_cost_min = df_eval['cost'].idxmin(axis=0, skipna=True)

        return df_eval.iloc[idx_cost_min, :]

    def create_leaf(self, dataset):
        """ Création d'une feuille
        INPUT
           - dataset : dataset de la feuille à construire
        OUTPUT
           - leaf : la classe feuille créée avec les informations de notre dataset"""

        labels = dataset[self.target]
        pop = labels.shape[0]
        class_dict = list(map(lambda x: {x: 1}, labels))
        class_dict_sum = reduce(self.__add_dict__, class_dict)
        prediction = max(class_dict_sum.items(), key=operator.itemgetter(1))[0]
        proba = {k: v / pop for k, v in class_dict_sum.items()}

        return leaf(dataset, pop, class_dict_sum, prediction, proba)

class leaf:
    def __init__(self, dataset, pop, class_dict_sum, prediction, proba):
        self.dataset = dataset
        self.pop = pop
        self.class_dict_sum = class_dict_sum
        self.prediction = prediction
        self.proba = proba

class decision_tree:

    def __init__(self, target, dataframe, max_depth):
        self.max_depth = max_depth
        self.target = target
        self.dataframe = dataframe

    def __quanti_split__(self, feature, value, dataset):
        left = dataset[dataset.loc[:, feature] <= value]
        right = dataset[dataset.loc[:, feature] > value]

        return left, right

    def __quali_split__(self, feature, value, dataset):
        left = dataset[dataset.loc[:, feature] == value]
        right = dataset[dataset.loc[:, feature] != value]

        return left, right

    def __add_dict__(self, prec_dict, new_dict):
        if list(new_dict.keys())[0] in prec_dict:
            prec_dict[list(new_dict.keys())[0]] += 1
        else:
            prec_dict[list(new_dict.keys())[0]] = 1
        return prec_dict

    def __gini__(self, dataset):
        rows = dataset[self.target]
        class_dict = list(map(lambda x: {x: 1}, rows))
        print("Here ", len(class_dict))
        class_dict_sum = reduce(self.__add_dict__, class_dict)
        occu_class = np.fromiter(reduce(self.__add_dict__,
                                        list(map(lambda x: {x: 1}, rows))).values(),
                                 dtype=float)
        pop = np.sum(occu_class)
        impurity = 1 - np.sum((occu_class / pop) ** 2)
        return impurity

    def __split_evaluator__(self, left_dataset, right_dataset):
        print('Left', left_dataset)
        left_eval = self.__gini__(left_dataset)
        nb_left = left_dataset.shape[0]
        print('Right', right_dataset)
        right_eval = self.__gini__(right_dataset)
        nb_right = right_dataset.shape[0]
        nb_tot = nb_left + nb_right
        cost = nb_left / nb_tot * left_eval + nb_right / nb_tot * right_eval
        return cost

    def __test_quali__(self, dataset, feature):
        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        for value in dataset.loc[:, feature].unique():
            left, right = self.__quali_split__(feature, value, dataset)
            cost_result = self.__split_evaluator__(left, right)
            df_eval = df_eval._append(pd.DataFrame([[feature,
                                                    value,
                                                    'quali',
                                                    cost_result]],
                                                  columns=('feature', 'value', 'nature', 'cost')))
        return df_eval

    def __test_quanti__(self, dataset, feature):
        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        value_to_test = (dataset.loc[:, feature].sort_values()[1:].values + dataset.loc[:, feature].sort_values()[
                                                                            :-1].values) / 2
        for value in value_to_test:
            left, right = self.__quanti_split__(feature, value, dataset)
            cost_result = self.__split_evaluator__(left, right)
            df_eval = df_eval.append(pd.DataFrame([[feature,
                                                    value,
                                                    'quanti',
                                                    cost_result]],
                                                  columns=('feature', 'value', 'nature', 'cost')))
        return df_eval

    def __find_best_split__(self, dataset):
        df_eval = pd.DataFrame([], columns=('feature', 'value', 'nature', 'cost'))
        columns = dataset.columns[np.logical_not(dataset.columns == self.target)]
        for column in columns:
            if len(dataset[column].unique()) >= 10:
                df_eval = df_eval.append(self.__test_quanti__(dataset, column))
            elif len(dataset[column].unique()) > 1:
                df_eval = df_eval._append(self.__test_quali__(dataset, column))

        df_eval = df_eval.reset_index(drop=True)

        idx_cost_min = df_eval['cost'].idxmin(axis=0, skipna=True)

        return df_eval.iloc[idx_cost_min, :]

    def create_leaf(self, dataset):
        labels = dataset[self.target]
        pop = labels.shape[0]
        class_dict = list(map(lambda x: {x: 1}, labels))
        class_dict_sum = reduce(self.__add_dict__, class_dict)
        prediction = max(class_dict_sum.items(), key=operator.itemgetter(1))[0]
        proba = {k: v / pop for k, v in class_dict_sum.items()}

        return leaf(dataset, pop, class_dict_sum, prediction, proba)

    def training(self, dataset, depth=0):
        # Cette partie de code vérifie que le dataset peut encore être séparé
        no_more_split = True
        columns = dataset.columns[np.logical_not(dataset.columns == self.target)]
        for column in columns:
            if len(dataset[column].unique()) > 1:
                no_more_split = False

        # Si le dataset est pure, ou que la profondeur maximum est atteinte ou
        # que le dataset ne peut plus être séparé nous créons une feuille
        if len(dataset[self.target].unique()) == 1 or depth == self.max_depth or no_more_split:
            return self.create_leaf(dataset)

        # Recherche de la meilleur séparation
        split_eval = self.__find_best_split__(dataset)

        # Si le coût obtenu après séparation est moins bon le coût actuel,
        # création d'une feuille avec le dataset actuel
        if split_eval['cost'] >= self.__gini__(dataset):
            return self.create_leaf(dataset)

        # Séparation du dataset selon la nature de la variable choisie
        if split_eval['nature'] == 'quali':
            left_branch, right_branch = self.__quali_split__(split_eval['feature'], split_eval['value'], dataset)
        elif split_eval['nature'] == 'quanti':
            left_branch, right_branch = self.__quanti_split__(split_eval['feature'], split_eval['value'], dataset)

        # Entraînement récursif de la branche de gauche
        left_node = self.training(left_branch, depth + 1)

        # Entraînement récursif de la branche de droite
        right_node = self.training(right_branch, depth + 1)

        # On retourne la racine de l'arbre
        return node(split_eval['feature'],
                    split_eval['value'],
                    split_eval['cost'],
                    split_eval['nature'],
                    left_node,
                    right_node,
                    depth,
                    dataset.shape[0])

    def fit(self):
        return self.training(self.dataframe)

def print_tree(node, spacing=""):
    # Différents affichages si c'est une feuille
    if isinstance(node, leaf):
        print(spacing + "Predict", node.prediction)
        print(spacing + "Predict", node.proba)
        return

    # Affichage de la condition de la séparation
    print(spacing + node.__split__())

    # Dans le cas où la condition est vérifiée
    print(spacing + '--> True:')
    print_tree(node.left_branch, spacing + "  ")

    # Dans le cas où la condition n'est pas vérifiée
    print(spacing + '--> False:')
    print_tree(node.right_branch, spacing + "  ")

tree_classif = decision_tree('classification', dataframe_classif, 4)
tree_trained = tree_classif.fit()
print_tree(tree_trained)