# fit_process

Ce projet ajuste le modèle sur le processus de potentiel de membrane à partir de la
projection sur une base de sinus/cosinus (méthode de l'article).

On utilise uniquement :

- le jeu de données traité dans `./data/processed/exp1/`
- les sorties d'adjacence dans `./adjancy/outputs/`

Il utilise :

- `selection/selected.npy` pour les neurones sélectionnés
- `small/interval_2/adjacency_small.npy` pour la matrice de Hawkes réduite
- `small/interval_2/baseline_small.npy` si disponible

Le plus grand petit intervalle de temps est utilisé par défaut pour l'estimation.
