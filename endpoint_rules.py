"""保守、可审计的测定分类。未知分母/来源进入 review，不猜测缩放常数。"""
from __future__ import annotations
import math
import re

RULE_VERSION = '1.0.1'
SPECIES = {9606: 'human', 10116: 'rat', 10090: 'mouse', 9615: 'dog', 9541: 'monkey'}
SPECIES_NAMES = {'homo sapiens': 'human', 'rattus norvegicus': 'rat', 'mus musculus': 'mouse',
                 'canis familiaris': 'dog', 'canis lupus familiaris': 'dog', 'macaca fascicularis': 'monkey'}


def family(kind):
    t = str(kind or '').lower().strip()
    if t in {'fu', 'ppb', 'unbound', 'unbound plasma'}:
        return 'fu'
    if t in {'f', 'relative bioavailability', 'logit bioavailability'}:
        return 'F'
    if t.startswith('papp') or t in {'caco-2 papp', 'logpapp'}:
        return 'Papp'
    if t in {'vdss', 'vdss(app)', 'vdss/f', 'vd', 'volume distribution', 'vd/f', 'vd app', 'vd area'}:
        return 'VDss'
    if t in {'t1/2', 't1/2 beta', 't1/2 h', 't1/2 iv', 'terminal elimination t1/2',
             'terminal t1/2', 'plasma half life', 'plasma half-life', 'beta t1/2', 't1/2lambda'}:
        return 'Thalf'
    if t in {'cl', 'clint', 'clh', 'clh(app)', 'clt', 'clb', 'clp', 'clblood', 'cltot', 'cltotal', 'clearance',
             'clerance', 'actual clearance', 'blood clearance', 'plasma clearance', 'clearance_blood',
             'renal clearance', 'cl_renal', 'cl-renal', 'cl_biliary', 'cl/f', 'cl/f_obs', 'clt/f', 'clz/f',
             'cl(app)', 'cl free', 'clup', 'clu/f', 'cld', 'cld/f', 'clm', 'clr'}:
        return 'clearance'
    return None


def normalize_unit(unit):
    s = str(unit or '').lower().replace('μ', 'u').replace('µ', 'u').replace('−', '-').replace('·', '.')
    s = re.sub(r'\s+', '', s).replace('^', '').replace('**', '')
    s = s.replace('(', '').replace(')', '')
    for word in ['kg', 'mg', 'min', 'h', 'g', 's']:
        s = s.replace(word+'-1', word)
    return s.replace('.', '/')


def classify(record):
    """返回来源标签、规范值/界限和处置理由；relation 不等于 '=' 不做精确回归。"""
    desc = str(record.get('description') or '').lower()
    kind = str(record.get('standard_type') or '').lower().strip()
    fam = family(kind)
    org = str(record.get('assay_organism') or '').lower()
    species = SPECIES_NAMES.get(org, 'unknown')
    tax = record.get('assay_tax_id')
    if tax is not None and str(tax) not in {'', 'nan'}:
        by_tax = SPECIES.get(int(tax), 'unknown')
        if species == 'unknown':
            species = by_tax
        elif by_tax != 'unknown' and species != by_tax:
            species = 'conflict'
    unit = normalize_unit(record.get('standard_units'))
    relation = str(record.get('standard_relation') or '').strip()
    result = dict(endpoint=fam or 'unknown', species=species, system='unknown', route='unknown',
                  canonical_value=None, canonical_unit='', conversion_rule='', status='review', reason='',
                  relation=relation, lower_bound=None, upper_bound=None, pH=None)
    ph = re.search(r'\bph\s*[=:]?\s*(\d+(?:\.\d+)?)', desc)
    if ph:
        result['pH'] = float(ph.group(1))
    iv = bool(re.search(r'intravenous|\bi\.?v\.?\b', desc))
    oral = bool(re.search(r'\boral\b|\bp\.?o\.?\b', desc))
    result['route'] = 'IV+PO' if iv and oral else 'IV' if iv else 'PO' if oral else 'unknown'
    reasons = []
    def reject(why):
        reasons.append(why)
    val = record.get('standard_value')
    try:
        val = float(val)
        if not math.isfinite(val):
            raise ValueError()
    except (TypeError, ValueError):
        reject('missing_or_nonfinite_value')
        val = None
    if species in {'unknown', 'conflict'}:
        reject('species_' + species)
    if record.get('data_validity_comment'):
        reject('source_validity_flag')
    if str(record.get('potential_duplicate') or '0') in {'1', '1.0'}:
        reject('source_potential_duplicate')
    if relation not in {'=', '<', '<=', '>', '>='}:
        reject('unknown_relation')
    factor, complement = None, False
    microsome = 'microsom' in desc
    hepatocyte = 'hepatocyt' in desc
    vitro = microsome or hepatocyte or bool(re.search(r'in vitro|homogenate|s9 fraction', desc))
    if fam == 'fu':
        result.update(endpoint='fu', canonical_unit='fraction')
        result['system'] = 'plasma' if 'plasma' in desc or kind == 'unbound plasma' else 'serum' if 'serum' in desc else 'unknown'
        if result['system'] == 'unknown' or vitro:
            reject('not_confirmed_plasma_or_serum_binding')
        if re.search(r'\bblood\b', desc) and result['system'] == 'unknown':
            reject('blood_not_plasma')
        factor = 0.01 if unit == '%' else 1.0 if unit in {'', 'fraction', 'ratio', 'dimensionless'} and kind in {'fu', 'unbound', 'unbound plasma'} else None
        complement = kind == 'ppb'
        result['conversion_rule'] = '1-bound_fraction' if complement else 'unbound_fraction'
    elif fam == 'clearance':
        result.update(endpoint='CL', canonical_unit='L/h/kg')
        if vitro:
            result['endpoint'] = 'CLint'
            result['system'] = 'microsome' if microsome and 'liver' in desc else 'hepatocyte' if hepatocyte else 'other_in_vitro'
            result['canonical_unit'] = 'uL/min/mg_protein'
            if result['system'] != 'microsome':
                reject('not_liver_microsomal_protein_task')
            if 'intrinsic' not in desc and kind != 'clint':
                reject('not_confirmed_intrinsic_clearance')
            if unit in {'ul/min/mg', 'ul/min/mgprotein', 'ul/min/microsomalproteinmg'}:
                factor = 1.0
            elif unit in {'ml/min/mg', 'ml/min/mgprotein'}:
                factor = 1000.0
            elif unit in {'ml/min/g', 'ml/min/gprotein'} and (re.search(r'per (?:g|gram)\b.{0,20}protein', desc) or unit.endswith('protein')):
                factor = 1.0
            elif unit in {'ul/min/g', 'ul/min/gprotein'} and (re.search(r'per (?:g|gram)\b.{0,20}protein', desc) or unit.endswith('protein')):
                factor = 0.001
            if re.search(r'\bunbound\b|binding.corrected|\bclint,u\b', desc):
                result['system'] = 'microsome_unbound'
        else:
            result['system'] = 'systemic_iv'
            if not iv or oral:
                reject('not_confirmed_IV_only')
            if re.search(r'hepatic|renal|biliary|intrinsic|free clearance|unbound clearance', desc) or kind in {'clh', 'clint', 'cl_renal', 'cl-renal', 'cl_biliary', 'clr', 'cl free', 'clup'}:
                reject('not_total_systemic_clearance')
            if '/f' in kind or re.search(r'apparent|oral clearance', desc):
                reject('apparent_clearance')
            if kind in {'clb', 'clblood', 'blood clearance', 'clearance_blood'} or re.search(r'\bblood clearance\b', desc):
                result['system'] = 'systemic_iv_blood'
            factor = {'l/h/kg': 1, 'ml/min/kg': .06, 'ml/h/kg': .001, 'l/min/kg': 60}.get(unit)
        result['conversion_rule'] = 'dimensional_conversion_only'
    elif fam == 'Papp':
        result.update(endpoint='Papp', canonical_unit='1e-6_cm/s', system='unknown')
        caco = bool(re.search(r'caco[ -]?2', desc+' '+kind))
        ab = bool(re.search(r'apical.{0,25}basolateral|a\s*(?:to|[-=]>|[-–])\s*b\b', desc+' '+kind))
        ba = bool(re.search(r'basolateral.{0,25}apical|b\s*(?:to|[-=]>|[-–])\s*a\b', desc+' '+kind))
        if caco and ab and not ba:
            result['system'] = 'caco2_ab'
        else:
            reject('not_confirmed_Caco2_AB')
        if 'inhibitor' in desc+' '+kind or 'log' in kind:
            reject('modified_or_log_permeability')
        if species != 'human':
            reject('Caco2_species_not_human')
        factor = {'cm/s': 1e6, '10-6cm/s': 1, "10'-6cm/s": 1, '10^-6cm/s': 1, '1e-6cm/s': 1,
                  '10-6cm/sec': 1, "10'-6cm/sec": 1, 'nm/s': .1, 'cm/sec': 1e6}.get(unit)
        result['conversion_rule'] = 'permeability_to_1e-6_cm_s'
    elif fam == 'F':
        result.update(endpoint='F', canonical_unit='fraction', system='absolute_oral')
        if 'relative' in kind+' '+desc or 'logit' in kind or not oral or not ('absolute' in desc or iv):
            reject('not_confirmed_absolute_oral_F')
        factor = .01 if unit == '%' else 1 if unit in {'fraction', 'ratio', 'dimensionless'} else None
        result['conversion_rule'] = 'fraction'
    elif fam == 'VDss':
        result.update(endpoint='VDss', canonical_unit='L/kg', system='steady_state_iv')
        if not iv or oral or ('vdss' not in kind and not re.search(r'steady.state', desc)) or '/f' in kind:
            reject('not_confirmed_IV_VDss')
        factor = {'l/kg': 1, 'ml/kg': .001}.get(unit)
        result['conversion_rule'] = 'volume_per_kg'
    elif fam == 'Thalf':
        result.update(endpoint='Thalf', canonical_unit='h', system='terminal_iv')
        if vitro or not iv or oral or not re.search(r'terminal|elimination|beta|lambda', desc+' '+kind):
            reject('not_confirmed_IV_terminal_half_life')
        factor = {'h': 1, 'hr': 1, 'hours': 1, 'min': 1/60, 's': 1/3600}.get(unit)
        result['conversion_rule'] = 'time_hours'
    else:
        reject('unsupported_endpoint')
    if factor is None:
        reject('unknown_unit_or_normalization_basis')
    elif val is not None:
        value = val * factor
        if complement:
            value = 1 - value
            relation = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '=': '='}.get(relation, relation)
            result['relation'] = relation
        result['canonical_value'] = value
        if result['canonical_unit'] == 'fraction':
            if not 0 <= value <= 1:
                reject('fraction_out_of_range')
            if fam == 'fu' and value in {0, 1} and relation == '=':
                reject('fu_boundary_requires_assay_review')
        elif value <= 0:
            reject('nonpositive_value')
        if relation in {'<', '<='}:
            result['upper_bound'] = value
        elif relation in {'>', '>='}:
            result['lower_bound'] = value
    result['task_id'] = '__'.join([result['endpoint'], species, result['system']])
    result['reason'] = ';'.join(sorted(set(reasons)))
    result['status'] = 'review' if reasons else 'accepted' if relation == '=' else 'censored'
    result['rule_version'] = RULE_VERSION
    return result
