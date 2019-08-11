#
# Directory structure.
# #
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

