#
# Directory structure.
#
data/raw:
	mkdir -p $@

data/raw/all_matches.json: | data/raw
	curl -o all_matches.json http://www.datdota.com/api/matches?tier=premium
	mv all_matches.json data/raw/


#
# Code for pulling updated data from Datdota.
#
DATE = $(shell date +%Y-%m-%d)
TIERS = premium professional semipro
.PHONY: $(TIERS)
$(TIERS): | data/raw
	curl -o $@_matches.$(DATE).json http://www.datdota.com/api/matches?tier=$@
	gzip $@_matches.$(DATE).json
	mv $@_matches.$(DATE).json.gz data/raw/
.PHONY: update_matches
update_matches: $(TIERS)


#
# Compute some predictions.
#
$(addprefix fitted/,ti9_grp_new_params_by_map.win_probs.gz \
					ti9_grp_new_params_by_map.player_skills.gz \ ti9_grp_new_params_by_map.player_skill_vars.gz): \
		data/raw/premium_matches.2019-08-19.json.gz | fitted
	python src/scripts/backtest.py \
		data/raw/premium_matches.2019-08-19.json.gz \
		newton \
		ti9_grp_new_params_by_map \
		4856 \
		144 \
		--scale 1.25 \
		--logistic_scale 3.0 \
		--radi_prior_sd 3.0
	gzip $(addprefix ti9_grp_new_params_by_map.,win_probs player_skills player_skill_vars)
	mv $(addprefix ti9_grp_new_params_by_map.gz,win_probs player_skills player_skill_vars) fitted
