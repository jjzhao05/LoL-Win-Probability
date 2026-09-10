# Gold or Objectives? A Study on League of Legends Win Probability

## Abstract

In League of Legends, the most commonly cited measure of who is winning a game is gold
difference. This project examines whether this is true, or whether it is just a convenient
approximation. Two win probability models, a logistic regression baseline and an XGBoost
gradient boosted tree ensemble, are trained on a per minute snapshot of real ranked matches
drawn from a single game patch. Both models are then subjected to feature ablations that remove
gold, experience, and level features in one pass, objective and tower features in another, and
finally the objective group's two sub components, epic monsters and structures, separately, in
order to measure how much predictive power each feature group is actually responsible for.

The data collection process, feature engineering, and modeling setup used to conduct this test
are described below. The results and analysis will be filled in once the pipeline has been run
on real collected data, at which point this abstract will also report the headline AUC figures
for each ablation.

## Background

League of Legends is a multiplayer online battle arena (MOBA) game in which two teams of five
players, the blue side and red side, compete on a map to destroy the opposing team's nexus. Each
player controls a champion and earns gold and experience over the course of a match by defeating
minions, monsters, and enemy champions. Gold is spent on items that make a champion stronger, and
experience raises a champion's level, unlocking and strengthening their abilities. Beyond
fighting for gold and experience directly, teams can also fight over neutral map objectives, epic
monsters such as dragons and Rift Herald, and defensive structures such as towers, both of which
grant a team a material advantage when secured, as well as less tangible map control and tempo. A
match ends when one team destroys the other's nexus, which usually only becomes possible after a
team has worn down enough of the opposing team's towers to reach it.

In League of Legends, the most commonly cited measure of who is winning a game is gold
difference. It is shown in the overlays on broadcasts across nearly every level of play, from
professional matches to the replay interface in the game client itself, presented as a readout of
which team currently holds the advantage. The reasoning behind it is straightforward, as gold
purchases grant items, which in turn increase a champion's power, and greater power tends to win
fights, so the team with more gold is presumed to be the team that is winning. Under this
framing, every other event in a game, dragons, towers, Rift Herald, first blood, matters only to
the extent that it helps a team accumulate or protect a gold advantage. If this framing is
correct, a win probability model requires little more than the gold difference between the two
teams to perform well.

This project treats that assumption as a hypothesis to be tested rather than a fact to be
assumed, since it is rarely examined directly in casual analysis of the game.

## Problem Statement

Existing win probability coverage of League of Legends, from broadcast overlays to the in client
replay tool, treats gold difference as a sufficient stand in for who is winning, without
separately testing whether objective control contributes predictive value beyond what gold
already captures. This project addresses that gap directly by measuring, rather than assuming,
how much of a win probability model's predictive power actually depends on objective control
once gold, experience, and level are already accounted for.

The core question is:

> Does win probability in League of Legends reduce to gold, or does objective control carry
> predictive value independent of the gold it generates?

This question is difficult to answer because gold, experience, kills, and objectives are all
heavily correlated with each other in a real match. For example, a team that secures a dragon
has usually just won a fight, and a team that wins a fight usually gains gold and experience in
the same instant. Furthermore, teams that have accumulated more gold tend to also have more
objectives. This project uses feature ablation, retraining each model with a feature group
removed rather than reading its raw correlation with winning, specifically to work around this
confound.

The objective feature group itself is not uniform. Epic monsters such as dragons, Rift Herald,
Baron, and the elder dragon are secured through a fight and provide a temporary buff or a burst
of gold and experience, while towers and plates are structures whose loss is permanent and whose
absence changes what the map allows a team to do for the rest of the game, commonly described as
map control. To see whether one of these sub groups is driving the objective effect more than the
other, the ablation study also removes structures and epic monsters separately, in addition to
removing the full objective group together.

## Terminology

A **champion** is the character a player controls for the duration of a match. **Gold** and
**experience (XP)** are the two primary resources a champion accumulates, spent on items and
levels respectively, and are collectively referred to in this report as **economy** features. A
**dragon**, **Rift Herald**, and **Baron Nashor** are neutral epic monsters that grant a
temporary team wide buff when defeated, and the **elder dragon** is a stronger version of a
dragon available later in the match. These four are referred to collectively as **epic
monsters**. A **tower** is a defensive structure that must be destroyed to advance toward the
enemy nexus, and a **plate** is the gold reward for damaging, though not necessarily destroying,
a tower before the fourteen minute mark. Towers and plates are referred to collectively as
**structures**. Together, epic monsters and structures make up the **objective** feature group
that this project tests against the economy feature group. A **match snapshot** or **per minute
row** refers to one row of the dataset, representing the full state of both teams at a single
minute of a single match.

## Data Collection

Match data is obtained from the Riot Games API, drawn from the `RANKED_SOLO_5x5` queue. Matches
are stratified across ten skill brackets, Iron through Diamond, evenly split across their four
divisions, along with Master, Grandmaster, and Challenger as single apex brackets, with a target
of 1,000 matches per bracket. Matches shorter than 15 minutes are excluded from collection, since
games that end that quickly are almost always remakes or early surrenders rather than genuinely
decided games.

League of Legends regularly receives biweekly patches, and thus changes considerably from patch
to patch. Champion balance shifts, item costs and effects may be changed, as well as structural
elements such as tower plating or objective bounties. Collecting data across multiple patches
over an extended window would confound rank with patch, since different brackets would inevitably
complete collection at different times. To avoid this, data collection is restricted to a single
named patch (`14.18`).

For each collected match, the Riot API's match timeline is used to reconstruct the state of the
game at one minute resolution, including cumulative gold, experience, level, creep score, damage
dealt and taken, kills, deaths, assists, and objective and tower state (dragons, heralds, barons,
elders, plates, and each of the eleven individual towers), tracked separately for both teams and
broken down by role where available. Each row in the resulting dataset represents a snapshot of
both teams at one minute of one match, labeled with the eventual winner.

## Feature Engineering

The raw per team, per role statistics collected for each match snapshot are transformed into
model ready features before training. For each statistic tracked on both teams, gold, experience,
level, creep score, and each damage category, a blue side minus red side difference feature is
derived, along with a per minute rate version of that difference where relevant, such as
`gold_diff_per_min` and `xp_diff_per_min`. Gold is additionally expressed as a share,
`team_100_gold_share`, the blue side's fraction of the combined gold pool at that snapshot, since
a share is naturally bounded and comparable across different points in the game in a way a raw
difference is not.

Objective and structure counts are handled the same way. Each epic monster and each structure
category is converted into a blue side minus red side difference feature, for example
`dragon_diff` and `tower_diff`, and it is exactly these difference features, identified by name,
that the ablation study removes to build the no economy, no objectives, no structures, and no
epic monsters feature sets described in the Problem Statement.

## Pipeline

The project pipeline runs as a sequence of scripts, orchestrated end to end by `main.py`.
`collect_data.py` is run manually and separately, since it is a slow, rate limited pull against
the Riot API rather than something to rerun on every pipeline execution. Once `data/` is
populated, `main.py` runs `combine.py`, which merges the per bracket snapshot files into a single
dataset, `summary_stats.py`, which computes descriptive statistics including the feature
correlation check described in the Problem Statement, `xgboost_engineering.py`, which builds the
model ready feature table described above, `xgboost_train.py` and `logistic_regression_train.py`,
which train and grid search each model against a shared match level split, and
`probability_plot.py`, which renders per match probability trajectories for both models.
`ablation.py` is run separately after both models exist, since it retrains both models again for
every ablation variant and is comparatively slow.

## Modeling Methodology

Both models are evaluated on an identical held out test set. The split is done at the match
level, meaning that a full match and all of its per minute rows are assigned entirely to either
the training or test partition.

Logistic regression fits a linear decision boundary over standardized features with an L2
penalty. It cannot represent interactions between features or nonlinear effects, but its
coefficients are directly interpretable, since a feature's sign and magnitude indicate precisely
how much it changes the predicted log odds of a blue side win.

XGBoost is a gradient boosted ensemble of decision trees, capable of representing nonlinear
effects and interactions between features. It can capture patterns such as a dragon advantage
mattering more when the game is otherwise close, a relationship a linear model cannot.

Both models are trained on the same features and evaluated against the same held out test set
described above. Each is tuned using a small grid search, as shown in Figure 1.

| Model | Hyperparameters Searched | Best Configuration |
|---|---|---|
| Logistic Regression | inverse regularization strength `C` (0.001, 0.01, 0.1, 1.0, 10.0, 100.0), L2 penalty | to be determined once the pipeline is run on real data |
| XGBoost | `max_depth` (3, 4, 6, 8), `learning_rate` (0.01, 0.03, 0.05, 0.1), `subsample` (0.6, 0.7, 0.85, 1.0), `colsample_bytree` (0.7, 0.85, 1.0) | to be determined once the pipeline is run on real data |

*Figure 1. Hyperparameter grids searched for each model, selecting on validation log loss.*

Both models are evaluated using the same criteria, using overall AUC, log loss, accuracy, and
Brier score on the test set, as well as the same metrics broken out by game minute bucket,
allowing a model's performance to be examined separately for the early, middle, and late stages
of a game.

## Interactive Dashboard

A Streamlit dashboard, `app.py`, was built to make the results explorable rather than fixed to
the figures printed in this report. It has two pages. The first, Explore a Game, lets a reader
step through a real held out match and watch both models' predicted win probability move minute
by minute alongside tower and objective events, or build a custom macro scenario, such as a
given gold lead paired with a given tower count, and see both models' live predictions for it
side by side. The second, Findings, presents the ablation results themselves: the headline AUC
comparison across all five feature groups with bootstrap confidence intervals, the closeness
stratified breakdown, calibration curves, feature importance for both models, and per minute
bucket performance. The dashboard reads the same output files this report is built from, so once
the pipeline has been run, both stay in sync automatically.

## Results and Analysis

*This section, including the feature ablation comparison across all five feature groups (full,
no economy, no objectives, no structures, and no epic monsters), the bootstrap confidence
intervals on the AUC gap for each of those groups, and the closeness stratified breakdown for
both the full objective removal and the structures only removal, will be written up once the
pipeline has been run on real collected data.*

### Logistic Regression Baseline

*Pending: calibration curve, feature importance by coefficient, and per minute bucket
performance, once the pipeline has been run.*

### XGBoost Ensemble

*Pending: calibration curve, feature importance by gain, and per minute bucket performance, once
the pipeline has been run.*

### Cross Model Observations

*Pending: whether the two models agree on the size and direction of the gold versus objectives
asymmetry, and whether they agree on which individual features matter most, once the pipeline
has been run. Agreement between a linear model and a tree ensemble here is what would make the
headline finding a property of the game rather than an artifact of one model's architecture.*

## Limitations

The ablation study measures association after controlling for other features present in the
model, not a causal effect from a randomized experiment. A real match cannot be replayed with
its objectives removed, so "how much AUC survives without objectives" describes how much
independent predictive signal the removed features carried in this dataset, not what would happen
to a team's actual win probability if a dragon were somehow taken away from them.

The feature groups used for ablation are defined by a keyword match against feature names, gold,
xp, and level for economy, and dragon, herald, baron, elder, plate, and tower for objectives. This
is simple and fully transparent, but it also means any feature whose name does not contain one of
these keywords, or whose name coincidentally does, is grouped by name rather than by a deeper
semantic check.

Collinearity within each feature group is a real concern for this project, though not in the
place it might first appear. `team_total_gold_diff` and `gold_diff_per_min` are near duplicates
of each other at a fixed minute, and the same is true of `team_xp_diff` and `xp_diff_per_min`, so
neither model's individual coefficient or feature importance value for any one of these features
should be read as that feature's unique, independent contribution. Logistic regression's L2
penalty keeps the model numerically stable when features are this correlated, but it does not
restore a unique attribution, it can still split weight arbitrarily between two features that
carry nearly the same information. This is why the report's headline claim rests on ablating
whole feature groups rather than reading off any single feature's coefficient or importance
score, since removing an entire correlated group together sidesteps the question of which
feature within that group gets credit and instead asks the more robust question of how much the
group as a whole is worth. The same caution does not apply as strongly to the gold versus
objectives comparison itself, since gold and objective features are correlated with each other
but are not near duplicates of each other in the way `team_total_gold_diff` and
`gold_diff_per_min` are.

The dataset is restricted to a single patch, which keeps rank from being confounded with patch
but also means the results describe this patch, not every point in the game's balance history.
`RANKED_SOLO_5x5` is standard ranked five versus five play, so restricting to that queue is not a
limitation in the same sense, it is simply what "League of Legends" means for the purposes of
this project. Whether the same asymmetry between gold and objectives holds on other patches,
where champion balance, item costs, or objective bounties have been tuned differently, is outside
the scope of this project.

Finally, the custom scenario predictor in the interactive dashboard only accepts macro, team
level inputs. Every feature the models were trained on that is not exposed as a slider, all per
role breakdowns, minute to minute momentum, and damage or vision detail, is held at zero rather
than estimated, so its predictions are a simplified sketch of model behavior rather than a
faithful reproduction of what either model would predict for a real, fully observed game state.

## Conclusion

*Pending: once the pipeline has been run, this section will state directly whether win
probability in League of Legends reduces to gold or whether objective control carries
independent predictive value, summarize the size of that effect for both models, and note
whether it concentrates in close games or holds evenly across game states.*

## References

[1] Riot Games. (n.d.). Riot Games API. Retrieved from https://developer.riotgames.com/

[2] Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
https://doi.org/10.1145/2939672.2939785

[3] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine
Learning Research, 12, 2825-2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

[4] Streamlit Inc. (n.d.). Streamlit documentation. Retrieved from https://docs.streamlit.io/
