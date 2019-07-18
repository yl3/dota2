#
# Directory structure.
# #
data/raw:
	mkdir -p $@

data/raw/all_matches.json: | data/raw
	curl -o all_matches.json http://www.datdota.com/api/matches?tier=premium
	mv all_matches.json data/raw/
