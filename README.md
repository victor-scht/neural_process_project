# Projet signaux neuronaux

A partir de signaux neuronaux récoltés sur une tortue, l'objectif est d'aligner
un modèle stochastique sur le potentiel de membrane du neurone d'étude. Il s'agit d'un modèle de diffusion couplé à un processus de Hawkes multi-dimensionnel.

Objectifs :

- Déterminer les neurones adjacents qui ont un impact sur le potentiel de membrane du neruone étudié.
- Estimer les paramètres du modèle à partir du potentiel de membrane et du sous-réseau neuronal identifié.

## Code

Le code est organisé dans plusieurs répertoires pour réaliser les différentes tâches.

- `build_dataset` : Construire le dataset qui servira pour estimer les paramètres du modèle.
- `adjancy` : estimation du sous-réseau neuronal à partir de la construction d'une matrice d'adjacence.
- `fit_process` : Estimation non-paramétrique du modèle avec projection sur une base de cosinus/sinus (méthode de l'article de référence).
- `fit_process_2` : Estimation non-paramétrique du modèle avec projection sur une base de spline.
- `test_fit_process` : Validation de la méthode de l'article, pour un jeu de données controlé (données synthétiques).

Bien que la méthode d'utilisation est décrite ci-dessous, il y a des fichiers `README.md` dans chaque répertoire.

## Utilisation

### Build Dataset

À exécuter dans le terminal :

```
python ./build_dataset/build_dataset.py
python ./build_dataset/visualize_data.py
```

Ou alors exécuter les fichiers dans un éditeur de code (Ex : Vscode).

Si besoin, modifier le fichier `./build_dataset/config.py` pour sélectionner les fenêtres temporelles (entre autres).

Cela génère le jeu de données propre dans le repertoire `./data/processed/` sous format `.npy` : aligner les piques d'intensité du neurone d'études avec les neurones adjacents, supprimer les neurones inactifs et les tests non significatifs (sans pique d'intensité du neurone d'étude).
Des courbes et des informations sur les données sont également produites dans le repertoire `./build_dataset/outputs/`.

### Matrice d'adjacence

À exécuter dans le terminal :

```
python ./adjancy/01_compute_full_adjacency.py
python ./adjancy/02_analyze_central_row.py
python ./adjancy/03_compute_small_adjacencies.py
```

Ou alors exécuter les fichiers directement dans un éditeur de code (Ex : Vscode).

Si besoin, modifier le fichier `./adjancy/config.py`.

- `./adjancy/01_compute_full_adjacency.py` : générer la matrice d'ajacence complète dans `./adjancy/outputs/full/A_full.npy`.
- `./adjancy/02_analyze_central_row.py` : analyser l'influence sur le neurone centrale sous échelle logarithmique (util pour choisir le seuil dans la config qui permet de déterminer le sous-réseau).
- `./adjancy/03_compute_small_adjacencies.py` : générer la nouvelle matrice d'adjacence à partir du sous-réseau (sans prendre en compte l'influence du neurone centrale).

### Aligner le modèle

Pour tester la méthode de l'article sur des jeux de données synthétiques, exécuter dans le teminal :

```
python ./test_fit_process/estimation.py
```

Pour tester la méthode sur un potentiel de membrane,

```
python ./fit_process/run_fit_process.py
```

Pour tester la méthode avec projection sur une base de spline,

```
python ./fit_process_2/run_fit_process.py
```

Modifier les fichiers de config (`config.py`) si nécessaires (par exemple pour sélectionner un test, une fenêtre temporelle, un échantillonage...).

Les résultats sont dans les répertoires `outputs/` associés.
