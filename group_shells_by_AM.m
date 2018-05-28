function new_shells = group_shells_by_AM(old_shells)
    am = [old_shells.am];
    max_am = max(am);
    am_count = zeros(1, max_am + 1);
    for i = 1 : length(old_shells)
        am_count(am(i) + 1) = am_count(am(i) + 1) + 1;
    end
    am_offsets = [1 cumsum(am_count)+1];
    
    am_count = zeros(max_am + 1, 1);
    new_shells = old_shells;  % just for allocate the memory
    for i = 1 : length(old_shells)
        curr_shell_am = am(i);
        new_shell_pos = am_offsets(curr_shell_am + 1) + am_count(curr_shell_am + 1);
        am_count(curr_shell_am + 1) = am_count(curr_shell_am + 1) + 1;
        new_shells(new_shell_pos) = old_shells(i);
    end
end