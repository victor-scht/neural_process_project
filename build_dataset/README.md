# Pipeline

1. Construire le jeu de données traité à partir des fichiers bruts `.mat` et `.csv` :

```bash
python build_dataset.py
```

2. Générer les sorties de visualisation, les fichiers csv, le rapport et les graphiques :

```bash
python visualize_selected.py
```

# Ce que fait chaque script

- `build_dataset.py`
    - charge les données intra et extra brutes
    - nettoie les horodatages extracellulaires
    - filtre sur l'intervalle `[T_START, T_END)`
    - décale les temps de spikes pour que l'intervalle commence à 0
    - infère les spikes centraux à partir du potentiel de membrane
    - supprime les essais où le neurone cible n'a aucun spike
    - supprime les neurones sans spike dans l'intervalle
    - sauvegarde les tableaux traités `.npy` et les métadonnées
- `visualize_data.py`
    - charge les `.npy` traités
    - crée les fichiers csv demandés pour les diagnostics
    - produit les graphiques de style notebook :
        - raster
        - superposition membrane + spikes
        - nuage de points inférés vs extra
        - histogramme des erreurs de timing
        - erreur en fonction du temps
        - histogramme du score d'alignement par neurone
    - crée les graphiques à partir des fichiers csv de diagnostic
    - génère un rapport en markdown
