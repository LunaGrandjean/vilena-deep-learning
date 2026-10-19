# Expériences MLP de Viviam — résultats historiques

Les quatorze fichiers JSON/PNG de ce dossier sont repris sans modification de
la branche `vivi`, commit `d395106` (« MLP: dropout 0.5 run at 64px »).
Ils documentent les essais de Viviam avant l'intégration dans la branche `luna`.

Ce dossier est l'unique point d'entrée pour les résultats MLP :

```text
reports/mlp/
    *.json et *.png          Expériences originales de Viviam
    archives/luna_32px/      Ancien modèle de Luna, conservé comme preuve historique
    runs/<nom>/             Nouvelles expériences avec le MLP de Viviam intégré
```

Le code ne contient qu'une classe MLP active : celle de Viviam. Le dossier
d'archive conserve les résultats de l'ancienne implémentation, pas un second
modèle à utiliser.

| Expérience | Validation au minimum de perte | Test enregistré |
|---|---:|---:|
| 64 px, couches 256/128, sans dropout | 36,7 % | 31,1 % |
| 128 px, expérience initiale | 43,8 % | 40,9 % |
| 64 px, couches 256/128, dropout 0,5 | 44,7 % | Non évalué dans les fichiers fournis |

Le script original normalise les pixels avec moyenne/écart-type fixés à 0,5.
Son ancien calcul du partage donne 1 049 images train, 226 validation et 225
test pour 1 500 images. Il ne conserve pas de signature du contenu des images.
L'ancienne comparaison utilisait le MLP de Luna à 32 px et le partage
1 050/225/225. Ses preuves sont désormais dans `archives/luna_32px/` ; le
tableau historique reste dans `reports/comparaison_v1/comparison.csv`.

Le MLP actif reprend l'architecture et les options de Viviam dans
`src/wikiart/model.py` et `src/wikiart/train_mlp.py`, avec les checkpoints,
signatures, chemins de données, rapports et mesures CPU de la branche `luna`.
Les nouvelles exécutions utilisent le partage commun 1 050/225/225.
Les séries brutes, résumés et courbes annotées restent exportés.

Le lanceur complet choisit maintenant le profil **64 px, 256/128, dropout 0,5,
AdamW, weight decay 0, normalisation 0,5**. Ce profil doit être exécuté sous le
protocole commun avant d'annoncer une nouvelle comparaison à quatre modèles.
Le 44,7 % ci-dessus est un score de validation historique, pas son score test.

Le script accepte aussi `--image-size`, `--hidden1`, `--hidden2`, `--dropout`
et `--weight-decay` pour reproduire la démarche d'étude de Viviam.
Sans `--test`, aucune prédiction sur le test n'est calculée. Avec `--test`,
le meilleur checkpoint est évalué avec les diagnostics et le benchmark CPU.
Les nouveaux rapports sont créés dans `reports/mlp/runs/<réglages>/` par défaut,
ou dans le chemin explicite `--output-dir`, sans écraser ces archives.
