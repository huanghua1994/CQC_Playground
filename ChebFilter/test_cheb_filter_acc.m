function test_cheb_filter_acc(min_ev, max_occ, min_uocc, max_ev)
    clf;
    
    min_ev1_track = mcweeny_purif_track(min_ev, max_occ, min_uocc, max_ev);
    track1_len = length(min_ev1_track);
    plot(min_ev1_track, 'b-*'), grid on, hold on
    
    [k, g, min_ev1, max_occ1, min_uocc1, max_ev1] = cheb_filter_acc(min_ev, max_occ, min_uocc, max_ev);
    fprintf("Chebyshev filter degree = %d\n", k);
    
    min_ev2_track = mcweeny_purif_track(-max_ev1, -max_occ1, -min_uocc1, -min_ev1);
    track2_len = length(min_ev2_track);
    plot(min_ev2_track, 'r-*')
    fprintf('Reduced McWeeny iterations = %d\n', track1_len - track2_len);
    
    max_len = max(track1_len, track2_len);
    axis([1, max_len, 0.5, 1.01]);
end