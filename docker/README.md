# LTX-2.5 W4A8 — runner désagrégé (Soma)

Génération vidéo LTX-2.5 (transformer W4A8 INT4, 22B audio+vidéo) sur **une seule
RTX 4080 16 Go**, via désagrégation par rôle : `encoder → denoiser → decoder`
enchaînés dans un conteneur, hand-off `.bin`, **un seul composant résident à la
fois**. C'est ce qui débloque la 1080p sur 16 Go sans thrash.

Script : [`generated_optimized.sh`](./generated_optimized.sh) · agrégateur de perf :
[`soma_perf_report.py`](../python/sglang/multimodal_gen/runtime/disaggregation/soma_perf_report.py)

---

## Prérequis (une fois)

```bash
# racine du PVC modèle (à ré-exporter après reboot)
export PVC=/var/lib/rancher/k3s/storage/pvc-716616bd-568b-45a2-badd-1b1541e7d93f_soma_soma-video-ltx-cache
```

- **Image** : `docker.io/library/sglang-fork:spike` (fork + env bakés).
- **Modèle** : `$PVC/spike/ltx-2.5-local` → symlink vers `ltx-2.5-diffusers`.
  Les composants sont mutualisés dans `$PVC/ltx-components/` (symlink farm).
- **Transformer W4A8** : `$PVC/ltx/w4a8/ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors`.
- **Fork monté en RO** : `/home/heero/Projects/sglang/python/sglang` (edits live, aucun rebuild).

---

## Commande de base (standard)

```bash
sudo k3s ctr -n k8s.io run --rm --gpus 0 --memory-limit 32212254720 \
  --env HF_HUB_OFFLINE=1 \
  --env HOME=/data/spike --env HF_HOME=/data/huggingface \
  --env SOMA_FFN_CHUNK=1024 \
  --env SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB=14.5 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env OUT=/data/spike/out/optimized --env WIDTH=640 --env HEIGHT=480 --env FRAMES=121 \
  --env PROMPT="A colossal bioluminescent jellyfish drifts through a deep ocean trench, slow steady push." \
  --mount type=bind,src=$PVC,dst=/data,options=rbind:rw \
  --mount type=bind,src=/home/heero/Projects/sglang/python/sglang,dst=/sgl-workspace/sglang/python/sglang,options=rbind:ro \
  --mount type=bind,src=/home/heero/Projects/sglang/docker/generated_optimized.sh,dst=/generated_optimized.sh,options=rbind:ro \
  docker.io/library/sglang-fork:spike ltxopt bash /generated_optimized.sh
```

Le report de perf tombe automatiquement en fin de run (voir plus bas).

---

## Options (variables d'env)

| Variable | Défaut | Rôle |
|---|---|---|
| `WIDTH` / `HEIGHT` | `832` / `480` | résolution |
| `FRAMES` | `49` | nombre de frames (le **vrai** coût mémoire, pas la réso) |
| `STEPS` | `8` | steps de denoise — **figé à 8 sur le distillé** (override ignoré) |
| `SEED` | `42` | graine (même seed = A/B reproductible) |
| `PROMPT` | méduse | prompt |
| `OUT` | `/data/spike/out/optimized` | chemin de sortie (sans `.mp4`) |
| `SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB` | `14.5` | cap VRAM (laisse la marge au compositeur) |
| `SOMA_REPORT` | `1` | `0` = pas de report de perf |
| `SOMA_OFFLOAD_PROFILE` | `0` | `1` = mesure le temps copy-stream de l'offload (events CUDA) |
| `EXTRA_ENCODER` / `EXTRA_DENOISER` / `EXTRA_DECODER` | `""` | flags CLI ajoutés par rôle (override COMMON, cf. tuning) |

**Passthrough par rôle** : `EXTRA_*` est appendé après les args communs, donc
il **override** (argparse : le dernier gagne). C'est le levier de tuning sans rebuild.

---

## Recettes

### ⚡ Rapide (aperçu basse réso)

```bash
sudo k3s ctr -n k8s.io run --rm --gpus 0 \
  --env HF_HUB_OFFLINE=1 --env HOME=/data/spike --env HF_HOME=/data/huggingface \
  --env SOMA_FFN_CHUNK=1024 --env SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB=14.5 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env OUT=/data/spike/out/quick --env WIDTH=512 --env HEIGHT=320 --env FRAMES=49 \
  --env PROMPT="A colossal bioluminescent jellyfish drifts through a deep ocean trench, slow steady push." \
  --mount type=bind,src=$PVC,dst=/data,options=rbind:rw \
  --mount type=bind,src=/home/heero/Projects/sglang/python/sglang,dst=/sgl-workspace/sglang/python/sglang,options=rbind:ro \
  --mount type=bind,src=/home/heero/Projects/sglang/docker/generated_optimized.sh,dst=/generated_optimized.sh,options=rbind:ro \
  docker.io/library/sglang-fork:spike ltxquick bash /generated_optimized.sh
```

### ✨ Qualité — décodeur de diffusion  ✅ VIABLE avec NATTEN

> **Verdict (mesuré, corrigé) : chemin qualité pour la haute réso.** L'ancien
> verdict "impasse 45 Go / 300 s" était faux — les 300 s / 45 Go étaient le
> **fallback FlexAttention** (construction du masque de voisinage), PAS le
> décodeur. Avec **NATTEN** installé (`na3d`), le decode retombe à quelques
> secondes et une VRAM normale. Il fallait aussi un **fix dtype** (q/k promus
> fp32 par norm+RoPE alors que v reste bf16 → `na3d` exige un dtype uniforme ;
> réaligné sur v).
>
> Gain qualité : **léger** à 480p (sharpness à peine perceptible sur un visage
> en gros plan), mais il **croît avec la résolution** — plus le VAE compresse de
> hautes fréquences (1080p, 2k), plus le diff decoder a de matière à
> resynthétiser. C'est donc un **outil de rendu final haute-réso**, pas un défaut
> quotidien (coûte ~2× le decode + dépendance NATTEN).
>
> ⚠️ **Caveat 2k** : le tiling est **partiel** — `forward_stages_1_to_3`
> ([ltx_2_5_diffusion_decoder.py:977](../python/sglang/multimodal_gen/runtime/models/decoders/ltx_2_5_diffusion_decoder.py#L977))
> tourne sur le **volume entier** avant la boucle de tuiles ; seuls les stages 4+
> sont tuilés. À 2k, ces 3 premiers stages non-tuilés seront le mur VRAM
> (chantier : les tuiler ou les offloader). Le VAE tuile tout, d'où sa frugalité.
>
> Prérequis : `pip install natten` dans l'image (roue compilée cp312/cu13x).

Remplace le décodeur VAE par le décodeur génératif LTX (synthétise le haut-fréquence
→ piqué à résolution constante). **Même denoise, mêmes latents, seul le decode change.**
Prérequis modèle : voir [Setup du décodeur de diffusion](#setup-du-décodeur-de-diffusion).

```bash
# ... commande de base ...
  --env OUT=/data/spike/out/diffdec \
  --env EXTRA_DECODER="--use-diffusion-decoder" \
# ...
```

Si la VRAM du decoder approche 14,5 (conv 2048 canaux en espace pixel) :
```bash
  --env EXTRA_DECODER="--use-diffusion-decoder --layerwise-offload-components diffusion_decoder"
```

### 🖼️ Haute résolution

```bash
  --env WIDTH=1920 --env HEIGHT=1088 --env FRAMES=121
```
La réso est quasi gratuite sur clip court (le pic est borné par le DiT résident +
tiling VAE) ; c'est **FRAMES** qui coûte. Surveille `vram_MB` de l'**encoder** (Gemma,
le rôle le plus gourmand) dans le report.

### 🎛️ Tuning offload (garder des couches DiT résidentes)

```bash
  --env EXTRA_DENOISER="--dit-layerwise-resident-layers 40 --dit-offload-prefetch-size 2"
```
⚠️ **Mesuré inutile ici** : le denoise est **compute-bound**, pas transfer-bound.
Garder des couches résidentes réduit le trafic H2D (78 → 21 GiB) mais **n'accélère
pas** le ms/step (le streaming était déjà caché derrière le compute) et rapproche
du cap VRAM. À laisser à `0` (défaut) sur machine partagée / haute réso.

### 📊 Profiling offload (temps copy-stream)

```bash
  --env SOMA_OFFLOAD_PROFILE=1
```
Ajoute `copy-stream ms` par rôle dans la section OFFLOAD (events CUDA, opt-in car
touche le hot path). Sinon on n'a que le **volume** H2D (GiB), pas le **temps**.

### 🅱️ Mode dev (plafond qualité, lent)

Le distillé est figé à 8 steps. Pour que les steps redeviennent un levier :
```bash
  --env EXTRA_DENOISER="..."   # + --model-variant dev côté COMMON
```
Dev = `--model-variant dev` (route le transformer vers `transformer_full/`) **+**
CFG>1 **+** 25-30 steps → ~7× plus lent, régime différent. **Projet à part**, pas
un tweak. Voir la note dev en bas.

---

## Le report de perf

Tombe en fin de run (sauf `SOMA_REPORT=0`). Table lisible + `OUT.report.json`.

- **wall_s / infer_s / load_s** : `load_s = wall - infer` = coût de chargement des
  poids de ce seul rôle.
- **vram_MB / host_MB** : pics par rôle. Le **TOTAL prend le max**, pas la somme —
  c'est toute la démonstration disagg (un composant résident à la fois).
- **denoise** : `Nx mean ms (min-max)` — le compteur `N` **vérifie** si l'override
  de steps a pris (distillé → reste `8x`).
- **OFFLOAD** : config resident/streamed/prefetch + volume `H2D GiB` (et `copy-stream
  ms` si profiling). `H2D` = **trafic** cumulé RAM→VRAM, **pas** l'occupation.

Un `WARNING` s'affiche si un fichier de perf porte un rôle inattendu (fichiers croisés).

JSON bruts par rôle : `$(dirname $OUT)/perf_<pid>/{encoder,denoiser,decoder}.json`.

---

## Sorties & évaluation

- mp4 : `$OUT.mp4`
- report : `$OUT.report.json`
- **Toujours valider en extrayant une frame** (le mp4 présent ne prouve rien) :

```bash
ffmpeg -y -i $OUT.mp4 -vf "select=eq(n\,30)" -vframes 1 ${OUT}_f30.png
```

Pour un A/B propre (codec écarté), compare des **PNG**, pas les mp4 (le codec par
défaut sort à 50/100 ; `--output-quality maximum` = 100, gratuit en compute).

---

## Setup du décodeur de diffusion

Le composant `diffusion_decoder` est `native_only` et **absent** des builds
allow-list. Deux conditions pour que `--use-diffusion-decoder` marche :

1. **Poids présents** dans `$PVC/ltx-components/diffusion_decoder/`
   (`config.json` + `diffusion_pytorch_model.safetensors` ~834 Mo). Le dir modèle
   y pointe par symlink (`ltx-2.5-diffusers/diffusion_decoder -> /data/ltx-components/diffusion_decoder`).
   ```bash
   hf download Lightricks/LTX-2.5-Diffusers --include "diffusion_decoder/*" --local-dir /tmp/dd
   sudo cp /tmp/dd/diffusion_decoder/*.safetensors "$PVC/ltx-components/diffusion_decoder/"
   ```
2. **Déclaré** dans `model_index.json` (sinon hard-error même poids présents) :
   ```bash
   D=$PVC/spike/ltx-2.5-diffusers
   sudo python3 -c "import json;p='$D/model_index.json';d=json.load(open(p));d['diffusion_decoder']=['diffusers','LTX2VideoDiffusionDecoderModel'];json.dump(d,open(p,'w'),indent=2)"
   ```

Vérif : `sudo ls -l $PVC/ltx-components/diffusion_decoder` doit montrer le safetensors,
et `sudo grep diffusion_decoder $D/model_index.json` la classe.

---

## Notes

- **✅ Config workhorse = distillé + VAE + lossless.** Net, rapide, mémoire-safe.
  Le VAE rend déjà le détail fin (vérifié sur visage : pores, taches de rousseur,
  cils). Le "flou" perçu était **le contenu** (méduse = sujet doux), pas le modèle.
- **✅ Décodeur de diffusion = VIABLE avec NATTEN** (voir section dédiée). Le
  "45 Go / 300 s" était le fallback FlexAttention, pas le décodeur ; NATTEN +
  fix dtype → decode en secondes, VRAM normale. Gain qualité léger à 480p, qui
  **croît en réso** → chemin rendu final 1080p/2k. Caveat : stages 1-3 non tuilés.
- **Écarté sur cette machine** : mode dev (7× plus lent).
- **Codec** : `output_quality`/`--output-compression` étaient **inopérants** (libx264
  ignore le `quality`=`-qscale` d'imageio → CRF 23 figé). Le save est **hardcodé
  lossless (`-crf 0`)** dans le fork ([utils.py](../python/sglang/multimodal_gen/runtime/entrypoints/utils.py),
  2 chemins). Gain = couleurs, pas piqué. À recâbler sur `output_compression` un jour.
- **🐛 ORBE = sglang DÉTECTE un cgroup (cause racine, enfin isolée).** Ce n'est
  ni kubelet, ni le reclaim, ni la valeur du cap. Isolé empiriquement : même
  `memory.max` (30 Gio, enforced côté hôte), **seul le fait que l'offload *lise*
  le cap change tout** :
  - `--memory-limit` **sans** mount `/sys/fs/cgroup` → cgroup **non détecté**
    (mount conteneur masqué) → offload budgète contre la RAM libre → **toutes les
    couches pinned** → **net, quelle que soit la valeur.**
  - cgroup **détecté** (mount, ou **pod k8s où kubelet l'expose**) → `HostPinBudget`
    lit `available = min(RAM libre, cap − usage)` → budget serré → les couches qui
    "ne rentrent pas" partent sur le chemin **mapped** (`MappedLayerCourier`, mmap +
    collecte **async**) → ce chemin dégrade (poids pas prêts au calcul) → **orbe.**
  Le mécanisme est dans [layerwise_offload.py](../python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py)
  (`HostPinBudget` :380, `available_host_memory`, `_plan_layer_hosting`, `MappedLayerCourier` :197).
- **🛡️ Garde-fou mémoire = `--memory-limit 32212254720`** (30 Gio), **SANS mount
  cgroup**. OOM-kill le runaway (ex. diff decoder ~45 Go) au lieu de figer la box →
  worst case `Exit -9`, jamais un reboot. Enforce côté hôte ; l'offload ne le
  **détecte pas** (`... (no cgroup cap)` dans le log) → **c'est justement ce qu'on
  veut** : protection SANS déclencher l'orbe. **Ne JAMAIS** ajouter
  `--mount .../sys/fs/cgroup` en prod (ça rendrait le cap détectable → orbe).
  Cleanup orphelins post-OOM (SIGKILL ne déclenche pas `--rm`) :
  `for k in task container snapshot; do sudo k3s ctr -n k8s.io $k rm <id>; done`.
- **⚠️ Déploiement k8s (Soma)** : en vrai pod, **kubelet EXPOSE le cgroup** → sglang
  le détecte → **orbes reviennent**. Fix requis avant prod : (a) corriger le chemin
  mapped (`MappedLayerCourier`) pour qu'il soit correct/synchrone, ou (b) rendre
  l'offload aveugle au cgroup (toujours pinner, le cap host gère l'OOM), ou (c) pod
  limit assez haute pour que tout tienne en pinned (pas de mapped).
- **Steps distillé** : `--num-inference-steps` est **ignoré** (schedule sigma figé 8).
  Le compteur `denoise Nx` du report le confirme. Monter les steps = mode **dev**.
- **Dev** : `--model-variant dev` → `transformer_full/`, + CFG + 25-30 steps. Plafond
  qualité supérieur mais ~7× plus lent. `dev à 8 steps < distillé à 8 steps` (sous-convergé).
- **⚡ SageAttention = ~36% plus rapide sur le denoise (MESURÉ, clips longs).**
  A/B à 1920×1088×**121 frames** : FA 15424 ms/step → sage 9803 ms/step (−36%),
  qualité préservée. Le gain **croît avec la durée** (attention O(N²)) → négligeable
  sur du 49f, gros sur du long. Auto-activé par le `.sh` (`SAGE=1`) quand le paquet
  est sur le PVC (`/data/sage-pkgs`, SageAttention 2.2.0 compilé cu130/sm89) ;
  sinon sglang retombe **silencieusement** sur Flash Attention (vérifier le log :
  `Attention backends for transformer: fa` = Sage PAS actif). `SAGE=0` pour couper.
  Installation (une fois, sur un cluster neuf) : `pip install --target=/data/sage-pkgs
  --no-deps --no-build-isolation git+https://github.com/thu-ml/SageAttention.git@d9704247…`
  dans un conteneur de l'image (a nvcc/torch/ninja), `--net-host` pour cloner.
- **`resident_layers`** = levier VRAM↔trafic, **pas** vitesse (le denoise est
  compute-bound côté matmuls W4A8 ; le streaming est déjà caché). Laisser à 0.

---

## Dépannage

| Symptôme | Cause | Fix |
|---|---|---|
| `Repo id must be in the form...` au chargement | `$PVC` non exporté → `/data` mal monté | `export PVC=...` |
| `appears incomplete (missing required components)` | composant déclaré dans `model_index.json` sans ses poids | copier les poids dans `ltx-components/<comp>/` |
| `does not declare a diffusion_decoder component` | `model_index.json` sans l'entrée | ajouter l'entrée (cf. setup) |
| mp4 sous un nom auto (prompt+timestamp) | ancien bug `output_file_name`/`perf_dump_path` qui transitait | corrigé dans le fork (exclusions transfert) |
| report `WARNING` rôle croisé / denoiser à 0 | ancien bug de dump intermédiaire | corrigé dans le fork |
```
