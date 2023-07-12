def seg_fn_to_unique_tma_code(fns):
    unique_codes=[]
    for fn in fns:
        core = 'Core[' + fn.split('Core[')[1].split(']_')[0] + ']'
        tma = 'TMA%s_' % fn.split(' #')[1][0:2]
        unique_codes.append(tma + core)
    return unique_codes