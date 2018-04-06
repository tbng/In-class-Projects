# plotting figure 3 in the review

from general_module import *


tpp_left_3, fdp_left_3 = asym_tpp_fdp(delta=0.5, epsi=0.15)
tpp_right_3, fdp_right_3 = asym_tpp_fdp(delta=0.3, epsi=0.15)

tpp_left_top_4_1, fdp_left_top_4_1 = asym_tpp_fdp(delta=1.0, epsi=0.1)
tpp_left_top_4_2, fdp_left_top_4_2 = asym_tpp_fdp(delta=1.0, epsi=0.2)
tpp_left_top_4_3, fdp_left_top_4_3 = asym_tpp_fdp(delta=1.0, epsi=0.4)

tpp_right_top_4_1, fdp_right_top_4_1 = asym_tpp_fdp(delta=1.0, epsi=0.2)
tpp_right_top_4_2, fdp_right_top_4_2 = asym_tpp_fdp(delta=0.8, epsi=0.2)
tpp_right_top_4_3, fdp_right_top_4_3 = asym_tpp_fdp(delta=0.4, epsi=0.2)

tpp_left_bottom_4_1, fdp_left_bottom_4_1 = asym_tpp_fdp(delta=0.1, epsi=0.03)
tpp_left_bottom_4_2, fdp_left_bottom_4_2 = asym_tpp_fdp(delta=0.1, epsi=0.05)
tpp_left_bottom_4_3, fdp_left_bottom_4_3 = asym_tpp_fdp(delta=0.1, epsi=0.07)

tpp_right_bottom_4_1, fdp_right_bottom_4_1 = asym_tpp_fdp(delta=0.25, epsi=0.05)
tpp_right_bottom_4_2, fdp_right_bottom_4_2 = asym_tpp_fdp(delta=0.1, epsi=0.05)
tpp_right_bottom_4_3, fdp_right_bottom_4_3 = asym_tpp_fdp(delta=0.05, epsi=0.05)

f, axarr = plt.subplots(3, 2, figsize=(16,18))

axarr[0, 0].plot(tpp_left_3, fdp_left_3)
axarr[0, 0].fill_between(tpp_left_3, fdp_left_3)
axarr[0, 0].set_ylim([0.0, 1.0])
axarr[0, 0].set_xlabel('TPP', fontsize=18)
axarr[0, 0].set_ylabel('FDP', fontsize=18)
axarr[0, 0].text(0.60, 0.1, 'Unachievable', fontsize=22, color='k')
axarr[0, 0].set_xlim([-0.01, 1])

axarr[0, 1].plot(tpp_right_3, fdp_right_3)
axarr[0, 1].fill_between(tpp_right_3, fdp_right_3)
axarr[0, 1].set_ylim([0.0, 1.0])
axarr[0, 1].set_xlabel('TPP', fontsize=18)
axarr[0, 1].text(0.60, 0.4, 'Unachievable', fontsize=22, color='k')
axarr[0, 1].set_xlim([-0.01, 1])

axarr[1, 0].plot(tpp_left_top_4_1, fdp_left_top_4_1)
axarr[1, 0].plot(tpp_left_top_4_2, fdp_left_top_4_2, '--')
axarr[1, 0].plot(tpp_left_top_4_3, fdp_left_top_4_3, '-.')
axarr[1, 0].set_xlim([-0.03, 1.05])
axarr[1, 0].set_ylim([-0.03, 1.0])
axarr[1, 0].set_xlabel('TPP', fontsize=18)
axarr[1, 0].set_ylabel('FDP', fontsize=18)
axarr[1, 0].legend(['$\epsilon = 0.1$', '$\epsilon = 0.2$', '$\epsilon = 0.4$'])

axarr[1, 1].plot(tpp_right_top_4_1, fdp_right_top_4_1)
axarr[1, 1].plot(tpp_right_top_4_2, fdp_right_top_4_2, '--')
axarr[1, 1].plot(tpp_right_top_4_3, fdp_right_top_4_3, '-.')
axarr[1, 1].set_xlim([-0.03, 1.05])
axarr[1, 1].set_ylim([-0.03, 0.7])
axarr[1, 1].set_xlabel('TPP', fontsize=18)
#axarr[1, 1].set_ylabel('FDP', fontsize=18)
axarr[1, 1].legend(['$\epsilon = 0.1$', '$\epsilon = 0.2$', '$\epsilon = 0.4$'])

axarr[2, 0].plot(tpp_left_bottom_4_1, fdp_left_bottom_4_1)
axarr[2, 0].plot(tpp_left_bottom_4_2, fdp_left_bottom_4_2, '--')
axarr[2, 0].plot(tpp_left_bottom_4_3, fdp_left_bottom_4_3, '-.')
axarr[2, 0].set_xlim([-0.03, 1.05])
axarr[2, 0].set_ylim([-0.03, 0.9])
axarr[2, 0].set_xlabel('TPP', fontsize=18)
axarr[2, 0].set_ylabel('FDP', fontsize=18)
axarr[2, 0].legend(['$\epsilon = 0.1$', '$\epsilon = 0.2$', '$\epsilon = 0.4$'])

axarr[2, 1].plot(tpp_right_bottom_4_1, fdp_right_bottom_4_1)
axarr[2, 1].plot(tpp_right_bottom_4_2, fdp_right_bottom_4_2, '--')
axarr[2, 1].plot(tpp_right_bottom_4_3, fdp_right_bottom_4_3, '-.')
axarr[2, 1].set_xlim([-0.03, 1.05])
axarr[2, 1].set_ylim([-0.03, 0.9])
axarr[2, 1].set_xlabel('TPP', fontsize=18)
#axarr[2, 1].set_ylabel('FDP', fontsize=18)
axarr[2, 1].legend(['$\epsilon = 0.1$', '$\epsilon = 0.2$', '$\epsilon = 0.4$'])

plt.tight_layout()
# plt.savefig('figure3-paper.pdf') # uncomment to save figure

plt.show()
plt.close('all')
