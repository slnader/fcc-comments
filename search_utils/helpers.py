def create_list_sequence(list_length, inc, include_last = True):
    """
    Creates a sequence of indices from a list of items
    Args:
    list_length: length of item list
    inc: increment of items to group together
    include_last: should the list lengths overlap by one
    """
    #Define row indices
    it_seq= range(0, list_length+1, inc)

    #Create tuples for each batch
    tuple_seq = []
    for i in range(len(it_seq)-1):
        if i ==0:
            tuple_seq.append((it_seq[i], it_seq[i+1]))
        else:
            if include_last:
                tuple_seq.append((it_seq[i], it_seq[i+1]))
            else:
                tuple_seq.append((it_seq[i]+1, it_seq[i+1]))

    if list_length%inc > 0:
        if include_last:
            tuple_seq.append((it_seq[i+1], it_seq[i+1]+list_length%inc))
        else:
            tuple_seq.append((it_seq[i+1]+1, it_seq[i+1]+list_length%inc))

    return tuple_seq
