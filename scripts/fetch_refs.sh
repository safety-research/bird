#!/usr/bin/env bash
# Fetch the primary sources the configs claim to reproduce.
#
# Three directories: refs/code/ upstream releases, refs/papers/*.pdf, and refs/tex/ the
# arXiv LaTeX source. None of them is distributed with this repository (see .gitignore);
# this script fetches them, is idempotent and never deletes.
# Prefer refs/tex/ over refs/papers/ when checking a quotation -- pdftotext mangles
# two-column layout and appendix prompt blocks, which is most of what the configs cite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/refs/code"
PAPERS="$ROOT/refs/papers"
TEX="$ROOT/refs/tex"
mkdir -p "$CODE" "$PAPERS" "$TEX"

clone() {  # clone <org/repo> <dest> [prune-path ...]
  local repo="$1" dest="$CODE/$2"; shift 2
  if [ -d "$dest" ]; then
    echo "  have  $(basename "$dest")"
    return
  fi
  echo "  clone $(basename "$dest")"
  git clone --depth 1 -q "https://github.com/$repo.git" "$dest"
  # PRUNE PATHS ARE THIRD-PARTY BENCHMARKS THE UPSTREAM VENDORED, never the
  # upstream's own work. CARD bundles whole copies of ManiSkill2 (75 MB) and
  # Metaworld (155 MB); its actual contribution -- code_generation/, rlkit/,
  # install/ -- is 216 KB, and both benchmarks are reachable on their own. Named
  # here rather than deleted by hand so a re-fetch is deterministic and the
  # omission is stated rather than discovered by someone wondering where
  # Metaworld went.
  # A prune path may carry a shell glob (`assets/meshes/link*`), expanded HERE,
  # relative to the fresh clone -- the caller quotes it so its own shell does not
  # expand it against the wrong directory. Needed for Assistax, whose
  # `assets/meshes/` mixes 33 MB of third-party Franka Panda meshes with 756 KB
  # of upstream's own bed and wheelchair meshes: a directory prune would have
  # taken upstream's work with the robot model, and the rule is third-party only.
  local prune path
  for prune in "$@"; do
    # An empty or slash-only prune argument would expand to "$dest"/ and the
    # rm -rf below would take the whole clone; refuse it rather than guard it
    # with ${path:?}, which cannot see that "$dest"/ is non-empty text.
    if [ -z "${prune//\/}" ]; then
      echo "        prune argument '$prune' is empty; refusing (it would name the clone itself)" >&2
      exit 2
    fi
    for path in "$dest"/$prune; do
      if [ -e "$path" ]; then
        echo "        prune ${path#"$dest"/} (third-party, vendored upstream)"
        rm -rf "${path:?}"
      fi
    done
  done
}

paper() {  # paper <arxiv-id> <name>
  local id="$1" name="$2"
  if [ -s "$PAPERS/$name.pdf" ]; then
    echo "  have  $name.pdf"
  else
    echo "  get   $name.pdf  (arXiv:$id)"
    curl -fsSL -A "bird-refs/0.1" "https://arxiv.org/pdf/$id" -o "$PAPERS/$name.pdf"
  fi
}

paper_url() {  # paper_url <url> <name>
  # For a paper that is NOT on arXiv. `paper()` builds an arXiv URL from an id;
  # a venue-hosted PDF has no id, and hand-adding the file (singh_orp's route)
  # leaves nothing that a re-fetch can check. This keeps the URL in the script,
  # where it is re-checkable.
  local url="$1" name="$2"
  if [ -s "$PAPERS/$name.pdf" ]; then
    echo "  have  $name.pdf"
  else
    echo "  get   $name.pdf  ($url)"
    curl -fsSL -A "bird-refs/0.1" "$url" -o "$PAPERS/$name.pdf"
  fi
}

# LaTeX source, and it is not a convenience -- it is the difference between a
# checkable quotation and a plausible one. `pdftotext` on a two-column paper
# breaks ligatures, bleeds columns into each other and silently mangles exactly
# the material the configs cite hardest: verbatim prompt blocks in appendices.
# Whether Gran Turismo's Appendix C says "Do not say which is better" is settled
# by the source in one grep -- the string does not exist and
# `gt_reward_design/neurips_2025.tex:690` says "Describe and compare in detail". Source also gives NAMED locators
# (`appendix.tex:352 \subsection{Reward Reflection}`) that survive a renumbering,
# where "App. §7.5" does not.
#
# Text only: .tex/.bbl/.bib/.sty/.cls and no figures. The text is a few MB
# against ~90 MB of tarballs, and a figure is not something a citation can be
# checked against -- refs/papers/*.pdf already carries them rendered.
#
# The fetch takes arXiv's CURRENT version of each paper. A `file:line` locator in
# a config comment was taken against one version, so an author's later revision
# can shift a cited line while the locator still looks precise; when one does not
# resolve, check it against the paper's earlier versions
# (`https://arxiv.org/e-print/<id>v<N>`). Refresh a paper deliberately by deleting
# its directory.
tex() {  # tex <arxiv-id> <name>
  local id="$1" name="$2" dest="$TEX/$2" tmp
  if [ -d "$dest" ] && [ -n "$(find "$dest" -name '*.tex' 2>/dev/null)" ]; then
    echo "  have  $name/"
    return
  fi
  tmp="$(mktemp -d)"
  if ! curl -fsSL -A "bird-refs/0.1" "https://arxiv.org/e-print/$id" -o "$tmp/src"; then
    echo "  MISS  $name (arXiv:$id) -- e-print unavailable"; rm -rf "$tmp"; return
  fi
  mkdir -p "$tmp/x"
  # arXiv serves either a tarball or a single gzipped .tex, and the second case
  # is not an error path -- several of these papers are one file.
  #
  # Decompress to a SCRATCH file and move it into place only on success. The
  # obvious `gunzip -c "$tmp/src" > "$tmp/x/main.tex"` cannot work: the shell
  # creates the target by redirection BEFORE gunzip runs, so a body that is
  # neither a tarball nor gzip -- a PDF-only submission, exactly the case the
  # message below names -- leaves a 0-byte `main.tex`. That satisfies the `*.tex`
  # probe, makes the PDF-only branch dead code, and copies an empty file into
  # $dest, after which the `have` check at the top of this function reports the
  # paper fetched FOREVER. A failed `tar` can also leave a partial extraction
  # behind (GNU tar extracts, then exits non-zero), so clear it before falling
  # back rather than mixing half a tree with a decompressed blob.
  # Branch on WHAT TAR PRODUCED, not on its exit status alone. `tar -xzf` on a
  # gzipped file that is not a tar archive exits 0 and extracts nothing when the
  # decompressed payload is under 512 bytes -- one tar block -- and exits 2 at or
  # above it (measured: 511 -> 0, 512 -> 2). So exit status alone is a predicate
  # that changes answer with the size of the paper, which is not a property
  # anything here should depend on.
  #
  # `-A` in the emptiness test is load-bearing: with plain `ls`, a tarball whose
  # members are all dot-prefixed reads as empty, drops into the fallback, and
  # `gunzip` on a valid tarball SUCCEEDS -- writing raw tar blocks out as
  # `main.tex`.
  if ! tar -xzf "$tmp/src" -C "$tmp/x" 2>/dev/null || [ -z "$(ls -A "$tmp/x" 2>/dev/null)" ]; then
    rm -rf "$tmp/x"; mkdir -p "$tmp/x"
    # The payload must be non-empty, contain no NUL, and not be a PDF. Two
    # narrow tests rather than one broad one (`grep -qI .`), because the broad
    # one ACCEPTS a small gzipped PDF: `%PDF-1.5` followed by ~2 KB of ASCII has
    # no NUL and no invalid byte, so grep calls it text and it lands as
    # `main.tex`. Measured against GNU grep 3.11. That is a regression the magic
    # check did not have, and it is the whole reason for the pair below.
    #
    #   * the NUL test catches the case a magic number cannot: a valid gzip
    #     whose payload is a tar archive GNU tar refuses while extracting
    #     nothing, which otherwise lands 10 KB of NUL-padded blocks as `main.tex`.
    #     WHOLE file, not a head window: measured 5.5 ms vs 11.5 ms on a 5 MB
    #     payload against a largest-real-payload of 124 KB, so the window bought
    #     nothing and cost the one thing a bounded scan always costs -- text for
    #     the first N bytes and garbage after is the silent-plausible-answer
    #     shape, in files that config comments cite by `file:line`.
    #   * `%PDF` catches a gzipped PDF, which contains no NUL when small -- so
    #     the NUL test alone is not enough (measured: a gzipped `%PDF-1.5` body
    #     is pure ASCII and passes it). Exactly four bytes. A `.tex` whose first
    #     four bytes really are `%PDF` is refused, which is a loud `MISS` and
    #     has never happened; the alternative is accepting every small PDF.
    if gunzip -c "$tmp/src" > "$tmp/one" 2>/dev/null && [ -s "$tmp/one" ] \
       && [ "$(wc -c < "$tmp/one")" -eq "$(tr -d '\000' < "$tmp/one" | wc -c)" ] \
       && [ "$(head -c 4 "$tmp/one")" != "%PDF" ]; then
      mv "$tmp/one" "$tmp/x/main.tex"
    fi
  fi
  if [ -z "$(find "$tmp/x" -name '*.tex' 2>/dev/null)" ]; then
    # An author may opt out of source distribution; arXiv then serves the PDF.
    echo "  MISS  $name -- no .tex in source (PDF-only submission)"; rm -rf "$tmp"; return
  fi
  mkdir -p "$dest"
  ( cd "$tmp/x" && find . \( -name '*.tex' -o -name '*.bbl' -o -name '*.bib' \
        -o -name '*.sty' -o -name '*.cls' \) -exec cp --parents {} "$dest/" \; ) 2>/dev/null
  echo "  get   $name/  ($(find "$dest" -name '*.tex' | wc -l | tr -d ' ') .tex, $(du -sh "$dest" | cut -f1))"
  rm -rf "$tmp"
}

echo "code:"
clone eureka-research/Eureka            Eureka
clone eureka-research/DrEureka          DrEureka
clone xlang-ai/text2reward              text2reward
clone google-deepmind/language_to_reward_2023 language_to_reward_2023
clone ShengjieSun419/CARD               CARD  ManiSkill2 Metaworld
clone Lossfunk/LIMEN                    LIMEN
# Nothing pruned: the checkout is 4.0 MB, of which 3.65 MB is `revolve.gif`, the
# paper's own demo animation. That is upstream's own work, and the rule above is
# that prune paths are third-party benchmarks the upstream vendored -- never this.
clone RishiHazra/Revolve                Revolve
# RF-Agent (NeurIPS 2025 spotlight) bundles whole copies of IsaacGymEnvs (312 MB)
# and rl_games (39 MB) beside its 5.5 MB contribution (RF_Agent/: the MCTS driver,
# the action prompts, the per-task generated rewards). Same argument as CARD's.
clone deng-ai-lab/RF-Agent              RF-Agent  isaacgymenvs rl_games
# LaRes (NeurIPS 2025) bundles a whole Metaworld copy (156 MB) beside its ~1.2 MB
# contribution (LaRes_from_scratch.py / LaRes_with_init.py, sac.py, utils/prompts/,
# rlkit/). Same argument as CARD's and RF-Agent's.
clone yeshenpy/LaRes                    LaRes  metaworld
# NOT a reproduced method: EPIC is the reward-distance pseudometric this repo
# reports candidates against (`evaluate.similarity.metric: epic`,
# `verify.quality_screen: epic`). Vendored for the same reason as the rest --
# a number we compute should be checkable against the construction it names.
clone HumanCompatibleAI/evaluating-rewards evaluating-rewards
# NOT a reproduced method either: FastTD3 (Seo et al., 2025) is the RL algorithm behind
# `train.backend: fasttd3` (bird/components/fasttd3.py, a line-faithful port) and the
# source of the published HumanoidBench per-env returns the h1hand anchors cite.
# Nothing pruned: the whole tree is ~830 KB and `data/` is upstream's own published
# training logs, not a vendored benchmark. Port pinned at upstream commit 229ed59;
# a re-fetch clones HEAD, so re-check the pin after re-fetching.
clone younggyoseo/FastTD3               FastTD3
# NOT a reproduced method: SimbaV2 (Lee et al., 2025, arXiv:2502.15280) is the RL
# algorithm behind `train.backend: simba_v2` (bird/components/simba_v2.py, a
# line-faithful torch port) and it is RDA's own HumanoidBench learner -- Table 1
# says "SAC" + "SimbaV2" (refs/tex/rda/appendix.tex:1420, :1421) and the official
# release IS that pair, headed "SAC with Hyper-Simba architecture"
# (configs/agent/simbaV2.yaml:2). Vendored so the port's file:line
# pins are checkable and so RDA's Table 1 can be checked against the config that
# produced it: two of its rows (gamma 0.98, 625K update steps) fall out of
# configs/online_rl.yaml's own formulas, which is the kind of claim that has to be
# re-derivable.
#
# NOTHING PRUNED, and that is the rule rather than an oversight: prune paths are
# third-party benchmarks the upstream vendored, never the upstream's own work, and
# SimbaV2's two big directories are its own -- `results/` (46 MB) is the published
# per-seed learning curves behind the paper's figures and `docs/` (23 MB) is its
# project page with its own videos. The clone is 72 MB, all of it upstream's.
# Apache-2.0 (LICENSE), which is the most permissive licence in refs/code/.
# Port pinned at upstream commit 86899c27 (2025-11-04); a re-fetch clones HEAD, so
# re-check the pin after re-fetching.
clone dojeon-ai/SimbaV2                 SimbaV2
# NOT a reproduced method: Assistax (Hinckeldey et al., RLJ 2026) is the assistive-robotics
# benchmark behind `problem.env_id: assistax_*` (bird/envs/assistax.py). Vendored
# so measured solver readings can be set against the published regime: the paper's
# solver settings, env counts and throughput claims have to be checkable here, in
# the tex and in the code that ran them. Pruned: the Franka Panda MuJoCo meshes (link*/hand*/finger*, 33 MB, a
# third-party robot model); upstream's own bed and
# wheelchair meshes (756 KB) stay. The decimated copies of every pruned mesh, with
# upstream sha256s, are already in bird/envs/assets/assistax/PROVENANCE.json.
# The paper and tex lines live in the benchmark block below; the arXiv id comes
# from the upstream README's header link,
# refs/code/assistax/README.md:4 -- the bibtex there names RLJ and no id.
clone assistive-autonomy/assistax       assistax  'assistax/envs/assets/meshes/link*' 'assistax/envs/assets/meshes/hand*' 'assistax/envs/assets/meshes/finger*'
# No public release: RDA, Gran Turismo. A "no release" line is a
# search someone did once -- a flag, not a fact. (RDA is a HALF exception: no
# framework source, but its project page publishes the generated reward code per
# task at https://nitinkamra1992.github.io/reward-design-agent/ .)

echo "papers:"
paper 2310.12931 eureka
paper 2406.01967 dreureka
paper 2309.11489 text2reward
paper 2306.08647 l2r
paper 2605.03408 limen
paper 2511.02094 gt_reward_design
paper 2410.14660 card
paper 2406.01309 revolve
paper 2602.23876 rf_agent
paper 2412.13492 roska         # AAAI 2025. The PDF as well as the source: the PDF is where
                               # the figures are (refs/tex/ is text only, by design).
paper 2006.13900 epic          # Gleave et al., ICLR 2021 -- the pseudometric, not a method
paper 2505.22642 fasttd3       # Seo et al., 2025 -- the algorithm behind train.backend: fasttd3
paper 2606.01672 rda           # RLC'26. Fetched from arXiv rather than added by hand, so a
                               # re-fetch re-checks it.
# R* (Li et al., ICML 2025, PMLR 267:34509-34527) is NOT on arXiv and has NO code
# release as far as a search found (arXiv full-text and listing search; GitHub
# code/repo search for "R* reward structure evolution", the author names, and the
# paper title). Treat those two absences like any "no release" line: a search
# someone did once, a flag rather than a fact.
# OpenReview (qZMLrURRr9) sits behind a browser challenge, so PMLR is the fetchable
# source.
paper_url https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25v/li25v.pdf rstar

# singh_orp.pdf (Singh, Lewis & Barto, 2009) predates routine arXiv posting and is added by
# hand.
# LaRes (Li et al., NeurIPS 2025) is fetched from the NeurIPS proceedings, not arXiv:
# no arXiv entry was found (arxiv.org listing search, and a site-restricted web search on
# the title + first author). That is a search someone did once -- re-check it rather than
# inheriting it. openreview.net serves
# the same PDF but sits behind a browser challenge that `curl` does not pass, so the
# proceedings URL is the one that re-fetches unattended.
paper_url https://proceedings.neurips.cc/paper_files/paper/2025/file/21b5d3a17aa5525f30bfd2bc59ac3a48-Paper-Conference.pdf lares

# Benchmark papers -- the environments BIRD runs on; provenance for the metric,
# expert and success-bar claims in captions and tasks/<id>/shared_spec.yaml. Papers + TeX
# only, never code: the environments are already vendored where BIRD uses them
# (`bird/envs/assets/`, the tier venvs, the pinned wheels the specs name).
paper 1910.10897 metaworld       # Yu et al., CoRL 2020 -- MT10/MT50, info["success"]
paper 2403.10506 humanoidbench   # Sferrazza et al., 2024 -- the `success_bar` ("Target") table
paper 2507.21638 assistax        # Hinckeldey et al., RLJ 2026 (upstream bibtex: journal={Reinforcement Learning Journal}) -- the five tasks, 52 wipe points
paper 2407.17032 gymnasium       # Towers et al., 2024; NeurIPS D&B 2025 -- the env interface the gym_* specs pin
# MuJoCo (Todorov, Erez, Tassa, IROS 2012) is not on arXiv and IEEE Xplore is paywalled.
# The URL every citation carries, homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf,
# now 301s to Todorov's relocated lab page (roboti.us/lab/), which links the same file
# under papers/ -- that is the fetchable location. No LaTeX source exists,
# so refs/tex/mujoco/ does not and will not; the venue is checkable against
# refs/tex/gymnasium/references.bib (`todorov2012mujoco`), not against the PDF itself.
paper_url https://roboti.us/lab/papers/TodorovIROS12.pdf mujoco

echo "sources (LaTeX, text only):"
tex 2310.12931 eureka
tex 2406.01967 dreureka
tex 2309.11489 text2reward
tex 2306.08647 l2r
tex 2605.03408 limen
tex 2511.02094 gt_reward_design
tex 2410.14660 card
tex 2406.01309 revolve
tex 2602.23876 rf_agent
tex 2412.13492 roska
tex 2006.13900 epic
tex 2606.01672 rda
# Benchmark papers (see the block above). mujoco has no arXiv source.
tex 1910.10897 metaworld
tex 2403.10506 humanoidbench
tex 2507.21638 assistax
tex 2407.17032 gymnasium

echo
echo "done. Not fetched by design: singh_orp (pre-arXiv -- the PDF is added by hand, there is"
echo "      no LaTeX source to get, so refs/tex/singh_orp/ does not and will not exist);"
echo "      rstar (PMLR-hosted, no arXiv entry, so refs/tex/rstar/ does not exist either --"
echo "      quotations from it are PDF-side and cannot carry a mechanical check);"
echo "      mujoco (IROS 2012, no arXiv entry -- fetched from roboti.us by paper_url, so"
echo "      refs/tex/mujoco/ does not exist and quotations are checked against the PDF)."
echo "Prefer refs/tex/ over refs/papers/ when checking a quotation -- see the tex() comment."
