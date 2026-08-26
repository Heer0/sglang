# SPDX-License-Identifier: Apache-2.0
"""Encodeur texte Gemma4-Unified (LTX-2.5), checkpoint comfy **int8 ConvRot sérialisé**.

Variante (b) « wrapper auto-géré » : au lieu de réimplémenter toute la pile texte
Gemma4-unified en couches TP quantifiées, on **réutilise tel quel le loader diffusers
prouvé en prod** (worker vidéo Soma, `sdk/soma_sdk/backends/ltx.py::_load_int8_encoder`) :

  1. squelette transformers `Gemma4UnifiedForConditionalGeneration` bâti sur `meta`
     (aucune allocation réelle) ;
  2. chaque Linear dont le poids est **int8** (avec sibling `.weight_scale`) est
     remplacé par `Int8ConvRotLinear` — le MÊME module que le worker : op
     `comfy_kitchen.int8_linear(..., convrot=True, convrot_groupsize=256)` ;
  3. les tenseurs du checkpoint (streamés par le loader SGLang) sont assignés
     directement dans les params/buffers (RAM ~= modèle int8, pas de spike bf16) ;
  4. les branches vision/audio/projection (jamais exécutées en texte-seul) restent
     matérialisées en zéros — juste assez pour que `.to(device)` ne casse pas.

`manages_checkpoint_quantization = True` : le lifecycle quant générique du loader
(`_configure_encoder_quantization` / `_require_quantized_encoder_layers` /
`_process_quantized_encoder_weights`) est **court-circuité** — on gère nous-mêmes le
placement des états quantifiés, comme l'encodeur Ideogram bitsandbytes.

Le `forward` **délègue** au sous-modèle texte transformers avec
`output_hidden_states=True` : LTX-2 empile les **49** hidden states (embedding +
48 couches) via `_gemma_postprocess_func` (`text_proj_in_factor=49`), donc la
numérique (embed_scale √hidden, k_eq_v, 4 RMSNorm, q/k-norm, layer_scalar =
`hidden_size_per_layer_input`, RoPE, sliding/full) reste 100 % celle de transformers.
On n'écrit AUCUN forward Gemma : seuls les Linear passent en int8.

Format quant confirmé (header du checkpoint) : 328 Linear I8[out,in] +
weight_scale F32[out,1], marker `{format:int8_tensorwise, convrot:true,
convrot_groupsize:256}`. Clés texte en `model.layers.*` (export transformers 5.10)
remappées `model.` -> `model.language_model.*` (refactor 5.15).
"""
from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# ConvRot groupsize du checkpoint (constant sur les 328 couches, cf. header).
_CONVROT_GROUPSIZE = 256


class Int8ConvRotLinear(nn.Module):
    """Linear int8 tensorwise + convrot (op comfy_kitchen). Copie conforme du module
    prouvé du worker diffusers (`backends/ltx.py`). Poids/scale = Parameters
    (requires_grad=False), assignés en streaming au chargement."""

    def __init__(self, in_features, out_features, bias, groupsize, device):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.groupsize = groupsize
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device), requires_grad=False)
            if bias
            else None
        )

    def forward(self, x):
        import comfy_kitchen as ck

        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        o = ck.int8_linear(
            x2,
            self.weight,
            self.weight_scale,
            self.bias,
            x.dtype,
            convrot=True,
            convrot_groupsize=self.groupsize,
        )
        return o.reshape(*shp[:-1], self.out_features)


def _remap(k: str) -> str:
    """Refactor transformers 5.10 -> 5.15 : le sous-modèle texte est déplacé sous
    `language_model`. Seules les branches texte sont remappées (chemin RELATIF au
    `Gemma4UnifiedForConditionalGeneration` transformers, i.e. `model.language_model.*`)."""
    if k.startswith("model.layers.") or k in (
        "model.embed_tokens.weight",
        "model.norm.weight",
    ):
        return "model.language_model." + k[len("model.") :]
    return k


class Gemma4UnifiedForConditionalGeneration(TextEncoder):
    """Encodeur texte LTX-2.5 int8-convrot, wrapper du modèle transformers.

    Nom de classe = nom d'architecture (`config.json::architectures`) pour que
    `ModelRegistry.resolve_model_cls(["Gemma4UnifiedForConditionalGeneration"])` le
    résolve (auto-scan AST des `EntryClass`)."""

    # On gère nous-mêmes les états quantifiés -> le lifecycle quant générique du
    # loader est court-circuité (pas de KitchenInt8 online, pas de post-process).
    manages_checkpoint_quantization = True
    supports_dp_encode = False
    layerwise_offload_dit_group_enabled = False
    # Chemin de la ModuleList des couches (relatif à CET encodeur), pour l'offload
    # layerwise éventuel : self.gemma.model.language_model.layers.
    layer_names = ["gemma.model.language_model.layers"]

    # Params non-texte (vision/audio/projection/lm_head) : jamais dans le corpus
    # texte, matérialisés en zéros -> tolérés manquants par le check de complétude
    # du loader. AUCUN motif texte ici, sinon un vrai trou de poids passerait muet.
    _allowed_missing_weights_patterns = [
        "vision",
        "audio",
        "projector",
        "multi_modal",
        "multimodal",
        "embed_vision",
        "embed_audio",
        "embedder",
        "tower",
        "siglip",
        "connector",
        "lm_head",
    ]

    def __init__(self, config) -> None:
        super().__init__(config)
        # bf16 imposé par le pipeline LTX-2.5 (text_encoder_precisions=("bf16",)).
        self._text_dtype = torch.bfloat16
        hf_cfg = self._resolve_hf_config(config)
        # Attn eager : pas de dépendance flash, coût négligeable (seq 1024).
        try:
            hf_cfg._attn_implementation = "eager"
        except Exception:
            pass
        self._hf_cfg = hf_cfg

        logger.info(
            "Gemma4Unified (int8-convrot, self-managed) : construction du squelette "
            "meta pour LTX-2.5. hidden=%s layers=%s",
            getattr(hf_cfg.text_config, "hidden_size", "?"),
            getattr(hf_cfg.text_config, "num_hidden_layers", "?"),
        )
        try:
            from accelerate import init_empty_weights
            from transformers import (
                Gemma4UnifiedForConditionalGeneration as _HFGemma4Unified,
            )

            # Squelette sur meta : structure seule, zéro allocation (l'énorme embed
            # 262144x3840 et les 48 couches ne sont jamais matérialisés en bf16).
            with init_empty_weights():
                self.gemma = _HFGemma4Unified(hf_cfg)
            self.gemma.eval()
        except Exception as exc:  # noqa: BLE001
            # Échec DUR (pas de fallback natif bf16 muet) : le loader re-raise
            # ComponentCheckpointUnsupportedError tel quel (component_loader ~l.274).
            from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (  # noqa: E501
                ComponentCheckpointUnsupportedError,
            )

            raise ComponentCheckpointUnsupportedError(
                f"Gemma4Unified int8: construction du squelette meta échouée: {exc}"
            ) from exc

        # Sous-module texte à interroger au forward (évite lm_head + routage
        # vision/audio). Résolu structurellement (hasattr, aucun forward).
        self._text = self._resolve_text_module()

    # ------------------------------------------------------------------ config
    def _resolve_hf_config(self, config):
        """Config transformers Gemma4Unified. Priorité au config.json du composant
        (exact), fallback sur le `gemma_config` embarqué dans le safetensors (le
        même que le worker diffusers). Échec explicite si aucun n'aboutit."""
        path = getattr(config, "_name_or_path", None) or getattr(
            config, "name_or_path", None
        )

        # 1. config.json du dossier composant (source SGLang standard).
        if path and os.path.isdir(path):
            try:
                from transformers import AutoConfig

                return AutoConfig.from_pretrained(path, trust_remote_code=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Gemma4Unified: AutoConfig.from_pretrained(%s) a échoué (%s), "
                    "fallback sur le gemma_config embarqué.",
                    path,
                    exc,
                )

        # 2. gemma_config embarqué dans le header du safetensors (proven diffusers).
        st_path = None
        if path:
            cand = path if path.endswith(".safetensors") else os.path.join(
                path, "model.safetensors"
            )
            if os.path.isfile(cand):
                st_path = cand
        if st_path:
            try:
                from transformers import Gemma4UnifiedConfig

                with open(st_path, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    hdr = json.loads(fh.read(n))
                meta = hdr.get("__metadata__", {})
                if "gemma_config" in meta:
                    return Gemma4UnifiedConfig.from_dict(
                        json.loads(meta["gemma_config"])
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Gemma4Unified: lecture du gemma_config embarqué (%s) échouée: %s",
                    st_path,
                    exc,
                )

        raise RuntimeError(
            "Gemma4Unified: impossible de résoudre la config transformers. "
            f"_name_or_path={path!r}. Attendu: un dossier composant avec config.json, "
            "ou un model.safetensors portant '__metadata__.gemma_config'."
        )

    def _resolve_text_module(self) -> nn.Module:
        """Sous-modèle texte (`self.gemma.model.language_model`), avec repli."""
        inner = getattr(self.gemma, "model", self.gemma)
        lm = getattr(inner, "language_model", None)
        return lm if lm is not None else inner

    def _locate(self, dotted: str):
        """Résout un nom de tenseur du checkpoint vers le param/buffer réel du modèle
        transformers, **sans présumer la version** : essaie le nom remappé (refactor
        5.15 `model.language_model.*`) PUIS le nom brut (export 5.10 `model.*`). Ceci
        rend le chargement robuste que l'image SGLang embarque un transformers ancien
        ou récent. Retourne (submodule, leaf, nom_relatif_encodeur) ou None."""
        gemma = self.gemma
        candidates = [_remap(dotted)]
        if dotted not in candidates:
            candidates.append(dotted)
        for cand in candidates:
            parent_path, _, leaf = cand.rpartition(".")
            try:
                submod = gemma.get_submodule(parent_path) if parent_path else gemma
            except AttributeError:
                continue
            if hasattr(submod, leaf) and (
                leaf in submod._parameters or leaf in submod._buffers
            ):
                return submod, leaf, "gemma." + cand
        return None

    # ------------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        out = self._text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(out, "hidden_states", None)
        last_hidden = getattr(out, "last_hidden_state", None)
        if last_hidden is None and hidden_states:
            last_hidden = hidden_states[-1]
        return BaseEncoderOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    # ------------------------------------------------------------- load_weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Assigne les tenseurs du checkpoint dans le squelette meta. Les Linear
        int8 (sibling `.weight_scale`) sont d'abord swappés en `Int8ConvRotLinear`,
        puis tous les poids sont posés en streaming. Retourne les noms (relatifs à
        CET encodeur) réellement chargés, pour le check de complétude du loader."""
        from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (  # noqa: E501
            ComponentCheckpointUnsupportedError,
        )

        dtype = self._text_dtype
        gemma = self.gemma

        # Matérialisation unique (le loader ne repasse pas l'itérateur). ~14 Go int8
        # + bf16 : tient sous le cap RAM du pod (38Gi).
        items = [(n, t) for n, t in weights]
        by_key = {n: t for n, t in items}

        # 1. Repérage des Linear quantifiés : weight int8 + sibling weight_scale.
        quant_prefixes: set[str] = set()
        for name, tensor in items:
            if name.endswith(".weight") and tensor.dtype == torch.int8:
                prefix = name[: -len(".weight")]
                if (prefix + ".weight_scale") in by_key:
                    quant_prefixes.add(prefix)

        # 2. Swap nn.Linear -> Int8ConvRotLinear. Localisation version-robuste (le
        #    module peut être sous `model.language_model.*` (transformers 5.15) ou
        #    `model.*` (export 5.10)).
        swapped = 0
        for prefix in quant_prefixes:
            loc = self._locate(prefix + ".weight")
            if loc is None:
                continue
            _, _, relname = loc  # "gemma." + <chemin réel du .weight>
            mod_path = relname[len("gemma.") : -len(".weight")]  # le Linear lui-même
            w = by_key[prefix + ".weight"]
            out_f, in_f = int(w.shape[0]), int(w.shape[1])
            has_bias = (prefix + ".bias") in by_key
            new_mod = Int8ConvRotLinear(
                in_f, out_f, has_bias, _CONVROT_GROUPSIZE, "meta"
            )
            parent_path, _, child = mod_path.rpartition(".")
            parent = gemma.get_submodule(parent_path) if parent_path else gemma
            setattr(parent, child, new_mod)
            swapped += 1

        # 3. Assignation en streaming de tous les tenseurs.
        loaded: set[str] = set()
        for name, tensor in items:
            if name.endswith(".comfy_quant") or name.startswith(
                ("hf_asset", "tokenizer_json")
            ):
                continue
            loc = self._locate(name)
            if loc is None:
                continue  # clé vision/audio/projection absente de l'arch -> ignorée
            submod, leaf, relname = loc
            t = tensor
            if t.is_floating_point():
                t = t.to(dtype)
            if leaf in submod._parameters:
                submod._parameters[leaf] = nn.Parameter(t, requires_grad=False)
            else:
                submod._buffers[leaf] = t
            loaded.add(relname)

        logger.info(
            "Gemma4Unified int8: %d Linear convrot swappés, %d tenseurs chargés "
            "(sur %d du checkpoint).",
            swapped,
            len(loaded),
            len(items),
        )
        # Garde-fou DUR : si aucun Linear int8 n'a été localisé, le mapping de noms
        # est cassé (mauvaise version transformers / arch) -> échec explicite plutôt
        # que fallback natif bf16 muet.
        if quant_prefixes and swapped == 0:
            raise ComponentCheckpointUnsupportedError(
                "Gemma4Unified int8: aucun des "
                f"{len(quant_prefixes)} Linear quantifiés n'a pu être localisé dans "
                "le modèle transformers (mapping de noms cassé). Exemple de clé: "
                f"{sorted(quant_prefixes)[0]!r}."
            )

        # 4. Buffers dérivés (jamais dans le ckpt) + tie lm_head <- embed_tokens.
        hidden_size = int(self._hf_cfg.text_config.hidden_size)
        for bname, buf in list(gemma.named_buffers()):
            if buf.is_meta and bname.endswith("embed_scale"):
                sm_path, _, _ = bname.rpartition(".")
                sm = gemma.get_submodule(sm_path) if sm_path else gemma
                sm._buffers["embed_scale"] = torch.tensor(
                    hidden_size**0.5, dtype=dtype
                )
        try:
            gemma.tie_weights()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemma4Unified: tie_weights a échoué: %s", exc)

        # 5. Meta résiduels (branches vision/audio non exécutées) -> zéros, sinon le
        #    .to(device) du loader plante sur « copy out of meta tensor ».
        for mod in gemma.modules():
            for store, is_param in ((mod._parameters, True), (mod._buffers, False)):
                for leaf, val in list(store.items()):
                    if val is not None and val.is_meta:
                        fdt = val.dtype if val.dtype.is_floating_point else dtype
                        z = torch.zeros(val.shape, dtype=fdt)
                        store[leaf] = (
                            nn.Parameter(z, requires_grad=False) if is_param else z
                        )

        gemma.eval()
        return loaded


EntryClass = Gemma4UnifiedForConditionalGeneration
