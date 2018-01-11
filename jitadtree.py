#coding:utf-8
import os, sys, re, argparse, time
import numba, numba.cuda
import numpy as np

@numba.njit('f8(f8[:,:],f8[:],i8,i8,f8,f8)')
def __update_fit(mat, backgrnd, i, j, delta, beta):
        """Fitting parameter
Parameters 
 mat : matrix for each chromosome
 backgrnd : array of background values
 i, j : index of location
 delta, beta : linear regression parameters
Returns
 value of fit (infinity is given if delta < 0)
 """
        n = j - i
        if delta < 0:
                return np.inf
        else:
                fit = 0
                for k in numba.prange(n-1):
                        for l in numba.prange(k+1,n):
                                fit += (((l-k) * delta + beta) * backgrnd[l-k] - mat[i+k,i+l]) ** 2
                return fit

@numba.njit('f8[2](i8, f8[:],f8[:])', parallel=True)
def __linregress(n, x, y):
        """Simpler version of linear regression which returns only slope and intercect.
Parameters
 x, y: values 
Returns
 numpy array of two float64 digits giving slope and intercect"""
        sx = sxx = sxy = syy = sy = 0
        for i in numba.prange(n):
                sx += x[i]
                sy += y[i]
                sxx += x[i] ** 2
                syy += y[i] ** 2
                sxy += x[i] * y[i]
        mx = sx / n
        my = sy / n
        vx = (sxx - mx * sx) / n
        vy = (syy - my * sy) / n
        cov = sxy / n - mx * my
        return np.array([cov / vx, my - cov * mx / vx])

# @numba.cuda.jit('void(i8, f8[:],f8[:],f8[2])', parallel=True)
# def __linregress_gpu(n, x, y, ret):
#         """Simpler version of linear regression which returns only slope and intercect.
# Parameters
#  x, y: values 
# Returns
#  numpy array of two float64 digits giving slope and intercect"""
#         sx = sxx = sxy = syy = sy = 0
#         for i in numba.prange(n):
#                 sx += x[i]
#                 sy += y[i]
#                 sxx += x[i] ** 2
#                 syy += y[i] ** 2
#                 sxy += x[i] * y[i]
#         mx = sx / n
#         my = sy / n
#         vx = (sxx - mx * sx) / n
#         vy = (syy - my * sy) / n
#         cov = sxy / n - mx * my
#         ret[0] = cov / vx
#         ret[1] = my - cov * my / vx
#
# @numba.cuda.jit('void(f8[:,:],f8[:],i8,i8,f8[:],f8[:],f8[3])')
# def __betadelta_gpu(mat, backgrnd, i, j, x, y, ret):
#         """Numba version of fitting function using linear regression. """
#         n = j-i
#         index = 0
#         for k in range(n):
#                 for l in range(k+1,n):
#                         x[index] = float(l-k)
#                         y[index] = mat[i+k,i+l] / backgrnd[l-k]
#                         index += 1
#         delta, beta = __linregress_gpu(index, x,y)
#         fit = __update_fit(mat, backgrnd, i, j, delta, beta)
#         ret[0] = beta
#         ret[1] = delta
#         ret[2] = fit

@numba.njit('f8[3](f8[:,:],f8[:],i8,i8,f8[:],f8[:])')
def __betadelta(mat, backgrnd, i, j, x, y):
        """Numba version of fitting function using linear regression. """
        n = j-i
        index = 0
        for k in range(n):
                for l in range(k+1,n):
                        x[index] = float(l-k)
                        y[index] = mat[i+k,i+l] / backgrnd[l-k]
                        index += 1
        delta, beta = __linregress(index, x,y)
        fit = __update_fit(mat, backgrnd, i, j, delta, beta)
        return np.array([beta,delta,fit])

@numba.njit('void(f8[:,:],f8[:],f8[:,:],f8[:,:],f8[:,:],i8,i8,f8[:],f8[:])')
def __update_matrices(mat, backgrnd, smat, gmat, bmat, n, height, x, y):
        for i in range(n-2):
                stop = n if n - i < height else height + i
                for j in range(i + 3, stop):
                        res = np.array([0., 0., 0.])
#                        __betadelta_gpu(mat, backgrnd, i, j, x, y, res)
#                        beta,delta,fit = res
                        # beta,delta,fit = __betadelta(mat, backgrnd, i, j, x, y)
                        beta,delta,fit = __betadelta(mat, backgrnd, i, j, x, y)
                        smat[i,j] = smat[j,i] = fit
                        gmat[i,j] = delta
                        bmat[i,j] = beta
                        

@numba.njit('void(f8[:,:],f8[:],f8[:,:],i8)', parallel=True)
def __update_smat(mat, backgrnd, smat, height):
        n = mat.shape[0]
        for i in numba.prange(n - 2):
                stop = n if n - i < height else height + i
                for j in numba.prange(i + 2, stop):
                        fit = 0
                        for k in numba.prange(j - i - 1):
                                for l in numba.prange(k + 1, j - i):
                                        fit += (mat[i + k, i + l] - backgrnd[ l - k ]) ** 2
                        smat[i, j] = smat[j, i] = fit
        return

@numba.njit('void(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:,:], f8[:,:], i8, i8, i8, i8, i8, f8[:,:])', parallel=True)
def __update_options(mat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, i, j, k, t, options):
        options[0,0] = local_score[k-1,t]
        olddelta = gmat[i,j]
        oldbeta = bmat[i,j]
        for l in range(min_size,k+1):
                stop = t if t < l - min_size else l - min_size
                for tt in range(stop):
                        if olddelta > gmat[i+k-l,i+k]:
                                options[l,tt] = np.inf
                        else:
                                oldscore = 0
                                for z in range(l-1):
                                        for w in range(z+1,l):
                                                oldscore += ((olddelta*float(w-z)+oldbeta)*backgrnd[w-z] - mat[i+k-l+z,i+k-l+w])**2                                                     
                                options[l,tt] = local_score[k-l,t-tt-1] + score[i+k-l,i+k,tt] - (oldscore - bakmat[i+k-l,i+k])


@numba.njit('void(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:,:], f8[:,:], i8, i8, i8, i8, i8, f8[:,:], i8[:,:], i8[:,:])')
def __update_local_traceback(mat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, t_lim, n, i, j, options, local_traceback_k, local_traceback_t):
        for k in range(min_size,n+1): 
                stop = t_lim if t_lim < k - min_size + 1 else k - min_size + 1
                for t in range(1,stop):
                        options[0:k+1,0:t] = 0
                        __update_options(mat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, i, j, k, t, options)
                        best = h = v = 0
                        for k_ in numba.prange(k+1):
                                for t_ in numba.prange(t):
                                        o_ = options[k_, t_]
                                        if o_ < best: best, h, v = o_, k_, t_
                        if h == 0 and v == 0: h, v = -1, t
                        local_score[k,t] = best
                        local_traceback_k[k,t] = h
                        local_traceback_t[k,t] = v

@numba.njit('i8(i8[:,:],i8[:,:],i8,i8,i8,i8,i8[:])')
def __update_intervals(local_traceback_k, local_traceback_t, i, n, t, min_size, intervals):
        """Numba version of interval update.
Parameters
 local_traceback_k, local_traceback_t : tracebacks
 i, n, t : position of interest
 min_size : minimum size of TAD
 interval_x, interval_y, interval_z ; tracing buffer
Returns
 length of intervals
"""
        pos = n
        tt = t
        index = 0
        while True:
                if pos <= min_size or local_traceback_k[pos,tt] == 0: break
                if local_traceback_k[pos,tt] == -1:
                        pos = pos - 1
                else:
                        newpos = pos-local_traceback_k[pos,tt]
                        intervals[index] = i + newpos
                        intervals[index + 1] = i + pos
                        intervals[index + 2] = local_traceback_t[pos, tt]
                        index += 3
                        newtt = tt - local_traceback_t[pos, tt] - 1
                        tt = newtt
                        pos = newpos
        return index


@numba.njit('i8(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:,:], f8[:,:], i8, i8, i8, i8, f8[:,:], i8[:], i8[:,:], i8[:,:], i8[:,:])' )
def  __update_local_partitions(mat, smat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, t_lim, height, L, options, intervals, local_traceback_k, local_traceback_t, partition_buffer):
        index_part = 0
        for n in range(min_size,height): # first line to last
                for i in range(L - n):       # first column to L-n th / type
                        j = i + n                # convert to diagonal
                        if smat[i,j] > 0: continue # skip if smat value is already more than 0
                        if gmat[i,j] < 0:          # set inf if gmat[i,j] is less than 0
                                score[i,j,:] = np.inf
                        else: 
                                score[i,j,0] = smat[i,j] # score[i,j,0] is negative
                                                        
                                local_t_lim = t_lim if t_lim < n - min_size + 1 else n - min_size + 1
                                local_score[0:n+1,0:local_t_lim] = 0.
                                local_traceback_k[0:n+1,0:local_t_lim] = 0
                                local_traceback_t[0:n+1,0:local_t_lim] = 0

                                __update_local_traceback(mat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, t_lim, n, i, j, options, local_traceback_k, local_traceback_t)
                                # print(np.sum(local_score[0:n+1,0:local_t_lim]), np.sum(local_traceback_k[0:n+1,0:local_t_lim]), np.sum(local_traceback_t[0:n+1,0:local_t_lim]))


                                for t in range(1,local_t_lim):
                                        itsize = __update_intervals(local_traceback_k, local_traceback_t, i, n, t, min_size, intervals)
                                        if itsize > 0:
                                                partition_buffer[index_part,0:4] = i,j,t,itsize
                                                partition_buffer[index_part,4:4+itsize] = intervals[0:itsize]
                                                index_part += 1

                                        score[i,j,t] = local_score[n,t] + smat[i,j]
        return index_part

# numba version of buildtree function
def __buildtrees(mat,backgrnd,smat,gmat,bmat,bakmat,t_lim,height,min_size):
        """Numba version of buildtree functions. This function consumed most of compuation time and __update_options function was introduced to enhance calculation."""
        # Reserve maximum size buffer to utilize enhancement by number in later processes
        L = smat.shape[0] # size of the matrix
        score = np.zeros((L,L,t_lim))
        local_parts_array = np.zeros((L,height,t_lim,t_lim,3),dtype=int)
        if height > L: height = L
        traceback = np.zeros((L,L,t_lim),dtype=int) # # LxLxt_lim tensor
        options = np.zeros((height + 1, height + 1))
        intervals = np.zeros(height * 3, dtype=np.int64)
        max_entries = height ** 2 * t_lim
        max_buffer = 4 + height * 3
        partition_buffer = np.zeros((max_entries, max_buffer), dtype=np.int64) 
        local_score = np.zeros((height + 1, t_lim + 1))
        local_traceback_k = np.zeros((height + 1, t_lim + 1), dtype=np.int64)
        local_traceback_t = np.zeros((height + 1, t_lim + 1), dtype=np.int64)

        # Build trees and backtrack matrices
        index_part = __update_local_partitions(mat, smat, gmat, bmat, bakmat, backgrnd, score, local_score, min_size, t_lim, height, L, options, intervals, local_traceback_k, local_traceback_t, partition_buffer)

        # Set traced paths in the original format
        for ind in range(index_part):
                i, j, t, l = partition_buffer[ind,0:4]
                buf = np.copy(partition_buffer[ind,4:4+l].reshape(l // 3, 3))
                local_parts_array[i, j - i, t, :l // 3, :] = buf
        return  local_parts_array, score


@numba.njit('void(f8[:,:],f8[:,:],f8[:,:,:],i8,i8,i8,i8,i8,i8[:,:],i8[:,:])')
def __update_forest_options(options, totalscore, score, min_size, L, T_lim, t_lim, height, traceback_k, traceback_t):
        for i in range(min_size, L):
                for t in range(1, T_lim):
                        options[0,0] = totalscore[i-1,t]
                        stop1 = height if height < i else i
                        for k in numba.prange(min_size,stop1):
                                stop2 = t if t < t_lim else t_lim
                                for tt in range(stop2):
                                        options[k,tt] = score[i-k,i,tt] + totalscore[i-k,t-tt-1]
                        h = v = 0
                        minval = options[0,0]
                        for h_ in range(i+1):
                                flag = False
                                for v_ in range(t):
                                        o_ = options[h_,v_]
                                        if o_ < minval:
                                                minval = o_
                                                h = h_
                                                v = v_
                        if h == 0 and v == 0:
                                h, v = -1, t
                        totalscore[i,t] = minval
                        traceback_k[i,t] = h
                        traceback_t[i,t] = v
        return
        
def getforest(score,L, height,T_lim,t_lim,min_size):
        totalscore = np.zeros((L,T_lim))
        traceback_k = np.zeros((L,T_lim),dtype=int)
        traceback_t = np.zeros((L,T_lim),dtype=int)
        options = np.zeros((L, T_lim))
        __update_forest_options(options, totalscore, score, min_size, L, T_lim, t_lim, height, traceback_k, traceback_t)
        return totalscore,traceback_k, traceback_t

@numba.njit('i8(i8[:,:],i8[:,:],i8,i8,i8,i8[:,:])',parallel=True)
def __foresttb(traceback_k, traceback_t, L, min_size, start_t, trees):
        pos = L-1
        tt = start_t
        index = 0
        while True:
                if pos <= min_size or tt == 0: break
                if traceback_k[pos,tt] == -1:
                        pos = pos - 1
                else:
                        newpos = pos - traceback_k[pos,tt]
                        trees[index] = newpos, pos, traceback_t[pos, tt]
                        index += 1
                        tt = tt - traceback_t[pos,tt] - 1
                        pos = newpos
        return index

        

def all_intervals(local_parts_array,i,j,t):
        intervals = [(i,j,t)]
        if t > 0:
                for p in local_parts_array[i,j-i,t]:
                        if np.sum(p) > 0:
                                intervals += all_intervals(local_parts_array,p[0],p[1],p[2])
        return intervals                                

def retrieve_parameters():
        """Read parameters from preference file or comman line argument
        """
        chrs = [] 
        paths = []
        N = []
        S = None
        p = None
        q = None
        M = None
        gamma = None
        output_directory = None

        default_filename_preferences = 'control_file.txt'
        default_output_directory = 'output'
        use_matrices_in_control_file = True
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', default=None, metavar='filename', help='control file : default first parameter or "control_file.txt" if exists')
        parser.add_argument('-S', type=int, metavar='number', default=None, help='max. size of TAD (in bins) : default 50')
        parser.add_argument('-M', type=int, metavar='number', default=None, help='max. number of TADs in each tad-tree : default 25')
        parser.add_argument('-p', type=int, metavar='number', default=None, help='boundary index parameter :  default 3')
        parser.add_argument('-q', type=int, metavar='number', default=None, help='boundary index parameter : default 12')
        parser.add_argument('-g', type=int, metavar='number', default=None, help='balance between boundary index and squared error in score function : default 500')
        parser.add_argument('-o', metavar='filename', default='output', help='output directory : default output')
        parser.add_argument('-i', default=[], metavar='directory', nargs='+', help='matrix files in plain text')
        parser.add_argument('--threads', type=int, metavar='number', default=1, help='number of processes (>=Python3.2): default 1')
        parser.add_argument('--quiet', action='store_true', help='Suppress verbose messages')
        parser.add_argument('control_file', nargs='*')
        args = parser.parse_args()

        verbose = not args.quiet
        num_threads = max(1, args.threads)
        if num_threads > 1:
                if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 2):
                        sys.stderr.write('Error : multiprocesssing is available only with >=Python3.2')
                        num_threads = 1
        contact_map_path = []
        contact_map_name = []
        N = []
        for fn in args.i:
                if fn.endswith('.txt'):
                        name = os.path.splitext(os.path.basename(fn))[0]
                        columns = 0
                        with open(fn) as fi:
                                for line in fi:
                                        items = line.strip().split('\t')
                                        if line.startswith('#'):
                                                m = re.search('name\\s*=\\s*(.*)', line)
                                                if m:
                                                        name = m.group(1)
                                        if line.startswith('#') is False and len(items) > 0:
                                                columns = len(items)
                                                break
                        if columns > 2:
                                # skip matrix files in control file if given in arguments
                                contact_map_path.append(fn)
                                contact_map_name.append(name)
                                N.append(columns)
                                use_matrices_in_control_file = False

        if len(args.control_file) > 0 and os.path.exists(args.control_file[0]):
                if args.i is None:
                        filename_preferences = args.control_file[0]
                else:
                        filename_preferences = default_filename_preferences
        elif args.i is not None and args.c is not None:
                filename_preferences = args.c
        else:
                filename_preferences = default_filename_preferences

        if len(contact_map_path) != len(N) or len(contact_map_name) != len(N):
                raise Exception('the numbers of name/filename/size should match')
        if os.path.exists(filename_preferences):
                matrix_sizes, contact_files, contact_names = [], [], []
                with open(filename_preferences) as fi:
                        for line in fi:
                                m = re.match('(\\w+)\\s*=\\s*(.*)', line)
                                if m:
                                        key = m.group(1)
                                        value = m.group(2).strip()
                                        if key == 'S': S = int(value)
                                        if key == 'p': p = int(value)
                                        if key == 'q': q = int(value)
                                        if key == 'M': M = int(value)
                                        if key == 'gamma': gamma = int(value)
                                        if key == 'output_directory': output_directory = value
                                        if use_matrices_in_control_file:
                                                if key == 'N': matrix_sizes = [int(x) for x in value.split(',')]
                                                if key == 'contact_map_path': contact_files = value.split(',')
                                                if key == 'contact_map_name': contact_names = value.split(',')
                if len(matrix_sizes) != len(contact_files) or len(contact_files) != len(contact_names):
                        raise Exception('the numbers of name/filename/size should match ({}/{}/{})'.format(len(matrix_sizes), len(contact_files), len(contact_names)))
                else:
                        for i in range(len(matrix_sizes)):
                                N.append(matrix_sizes[i])
                                contact_map_path.append(contact_files[i])
                                contact_map_name.append(contact_names[i])

        # remove duplicated or invalid data and normalize names
        for i in range(len(contact_map_name)):
                name = contact_map_name[i]
                fn = os.path.abspath(contact_map_path[i])
                if os.path.exists(fn) is False or N[i] <= 2:
                        if verbose: sys.stderr.write('{} was not found for {}\n'.format(fn, name))
                        contact_map_name[i] = contact_map_path[i] = N[i] = None
                        continue
                name_counter = 0
                duplicated = False
                for j in range(i):
                        if name == contact_map_name[j]:
                                if fn == os.path.abspath(contact_map_path[j]): # duplicate
                                        if verbose: sys.stderr.write('{} is duplicated\n'.format(fn))
                                        contact_map_name[i] = contact_map_path[i] = N[i] = None
                                        duplicated = True
                                        break
                                else:
                                        name_counter += 1
                if not duplicated and name_counter > 0:
                        while True:
                                name_ = '{}-{}'.format(name, name_counter)
                                if name_ not in contact_map_name[:i]:
                                        contact_map_name[i] = name_
                                        break
                                else:
                                        name_counter += 1
        contact_map_name = [x_ for x_ in  contact_map_name if x_ is not None]
        contact_map_path = [x_ for x_ in  contact_map_path if x_ is not None]
        N = [x_ for x_ in N if x_ is not None]

        # overload parameters in control file by given parameters
        if args.S is not None: S = args.S
        if args.p is not None: p = args.p
        if args.q is not None: q = args.q
        if args.M is not None: M = args.M
        if args.g is not None: gamma = args.g
        if args.o is not None: output_directory = args.o
        # set default values

        if S is None: S = 50
        if M is None: M = 25
        if p is None: p = 3
        if q is None: q = 12
        if gamma is None: gamma = 500
        if len(contact_map_path) == 0:
                raise Exception('no input files given')
        if output_directory is None:
                output_directory = default_output_directory

        if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        parameters = {'S':S, 'p':p, 'q':q, 'M':M, 'gamma':gamma, 
                'output_directory':output_directory, 
                'contact_map_path':contact_map_path, 'contact_map_name':contact_map_name, 'N':N,
                'threads':num_threads,
                'verbose':verbose
                }
        return parameters

def __process_chromosome(chr_index, min_size, t_lim, T_lim, mat, backgrnd, gmat, bmat, bakmat, smat, parameters):
        map_file = parameters['contact_map_path'][chr_index]
        map_name = chr = parameters['contact_map_name'][chr_index]
        map_size = parameters['N'][chr_index]
        output_directory = parameters['output_directory']
        p = parameters['p']
        q = parameters['q']
        gamma = parameters['gamma']
        verbose = parameters['verbose']
        S = height = parameters['S']
        M = parameters['M']
        num_threads = parameters['threads']
        short = p; long = 1; steps=q
        bi = []
        for i in range(long,mat.shape[0]-long):
                b = 0
                for s in range(1,steps):
                        a1 = np.sum(mat[i-long*s:i-long*(s-1),i-short:i])
                        b1 = np.sum(mat[i-long*s:i-long*(s-1),i:i+short])
                        a2 = np.sum(mat[i+long*(s-1):i+long*s,i-short:i])
                        b2 = np.sum(mat[i+long*(s-1):i+long*s,i:i+short])       
                        b += np.abs(a1-b1) + np.abs(a2-b2)      
                bi += [b]
        bi = [0]*long + bi + [0]*long
        bi = (np.array(bi) - np.mean(bi)) / np.std(bi) 

        for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                        if bi[i] < 0 or bi[j] < 0: smat[i,j] = np.inf
                        else: smat[i,j] = smat[i,j] - gamma*(bi[i]+bi[j])
        if verbose and num_threads == 1: 
                sys.stderr.write('\033[F\033[K')
                sys.stderr.write(time.strftime('[%H:%M:%S] Building TADtrees for ', time.localtime()) + chr + '\n')

        L = smat.shape[0] # size of the matrix
        local_parts_array, score = __buildtrees(mat, backgrnd, smat, gmat, bmat, bakmat, t_lim, height, min_size) # numba version
        
        if verbose and num_threads == 1: 
                sys.stderr.write('\033[F\033[K')
                sys.stderr.write(time.strftime('[%H:%M:%S] Assembling TADforest for ', time.localtime()) + chr + '\n')
        totalscore,traceback_k,traceback_t = getforest(score, L, height, T_lim, t_lim, min_size)

#                print(np.sum(mat[mat<1e10]), np.sum(smat[smat<1e10]), np.sum(gmat[gmat<1e10]), np.sum(bmat[bmat<1e10]),np.sum(bakmat[bakmat<1e10]))

        if not os.path.exists(os.path.join(output_directory, chr)):
                os.makedirs(os.path.join(output_directory, chr))
        duplicates = []

        for start_t in range(1,T_lim):
                L = totalscore.shape[0]
                trees = np.zeros((L * 2,3), dtype=np.int64)
                _tree_size = __foresttb(traceback_k, traceback_t, totalscore.shape[0], min_size, start_t, trees)#_tx, _ty, _tz)
                trees = trees[0:_tree_size]
                allints = []

                for t in trees:
                        allints += all_intervals(local_parts_array,t[0],t[1],t[2])

                allints = sorted(allints, key = lambda x: np.abs(x[1]-x[0]))

                final_ints = []
                for t in allints:
                        duplicate = False
                        for tt in final_ints:
                                if (np.abs(t[0] - tt[0]) < 2 and np.abs(t[1] - tt[1]) < 2):
                                        duplicate = True
                                        
                        if not duplicate:
                                final_ints.append([t[0], t[1]])

                with open(os.path.join(output_directory, chr, 'N{}.txt'.format(start_t)), 'w') as fo:
                        fo.write('chr\tstart\tend\n')
                        for t in sorted(final_ints, key=lambda x:x[0]):
                                fo.write('{}\t{}\t{}\n'.format(chr, t[0], t[1]))
                duplicates.append((start_t, 1. - float(len(final_ints)) / len(allints)))
                
        with open(os.path.join(output_directory, chr, 'parameters.txt'),'w') as fo:
                fo.write('Filename : {}\n'.format(map_file))
                fo.write('Size : {}\n'.format(map_size))
                fo.write('Name : {}\n'.format(map_name))
                fo.write('Maximum TAD size : {}\n'.format(S))
                fo.write('Maximum TADs in each tree : {}\n'.format(M))
                fo.write('Boundary index parameter : {} , {}\n'.format(p, q))
                fo.write('Balance between boundary : {}\n'.format(gamma))
                
        with open(os.path.join(output_directory, chr, 'proportion_duplicates.txt'),'w') as fo:
                fo.write('name\tproportion_duplicates\n')
                for val in duplicates:
                        fo.write('{}\t{}\n'.format(val[0], val[1]))
        if verbose:
                if num_threads == 1: 
                        sys.stderr.write('\033[F\033[K')
                sys.stderr.write(time.strftime('[%H:%M:%S] ', time.localtime()) + chr + ' completed\n')

def main():
        parameters = retrieve_parameters()
        S = parameters['S']
        p = parameters['p']
        q = parameters['q']
        M = parameters['M']
        gamma = parameters['gamma']
        output_directory = parameters['output_directory']
        contact_map_path = parameters['contact_map_path']
        contact_map_name = parameters['contact_map_name']
        N = parameters['N']
        verbose = parameters['verbose']
        num_threads = parameters['threads']

        if verbose:
                sys.stderr.write('Maximum TAD size : {}\n'.format(S))
                sys.stderr.write('Maximum TADs in each tree : {}\n'.format(M))
                sys.stderr.write('Boundary index parameter : {} , {}\n'.format(p, q))
                sys.stderr.write('Balance between boundary : {}\n'.format(gamma))
                sys.stderr.write('Output directory : {}\n'.format(output_directory))
                sys.stderr.write('Contact files : {}\n'.format(','.join(contact_map_path)))
                sys.stderr.write('Contact names : {}\n'.format(','.join(contact_map_name)))
                sys.stderr.write('Contact sizes : {}\n'.format(','.join(['{}'.format(n) for n in N])))
                sys.stderr.write('Number of threads: {}\n'.format(num_threads))

        #----------------------------------------------------------------------------------------#
        #                                                          LOAD CONTACTS AND BACKGROUND
        #----------------------------------------------------------------------------------------#
        # load data
        if verbose: sys.stderr.write(time.strftime('[%H:%M:%S] Loading data        \n', time.localtime()))
        chrs = contact_map_name
        paths = contact_map_path

        mats = {chrs[i] : np.loadtxt(paths[i]) for i in range(len(paths))}
        height = S
        backbins = []
        for chr in chrs:
                for i in range(mats[chr].shape[0]-height):
                        backbins += [mats[chr][i,i:i+height]]
        backgrnd = np.mean(backbins,axis=0)

        tadscores = {}
        bakscores = {}
        chrdeltas = {}
        chrbetas = {}

        # Parameters precomputation
        for chr_index in range(len(chrs)):
                chr = chrs[chr_index]
                if verbose: 
                        if chr_index > 0:
                                sys.stderr.write('\033[F\033[K')
                        sys.stderr.write(time.strftime('[%H:%M:%S] Precomputing paramters for ', time.localtime()) + chr + '\n')
                n = mats[chr].shape[0]
                smat = np.zeros((n,n))
                gmat = np.zeros((n,n))
                bmat = np.zeros((n,n))
                x = np.ndarray(n ** 2 // 2, np.float64)
                y = np.ndarray(n ** 2 // 2, np.float64)
                __update_matrices(mats[chr], backgrnd, smat, gmat, bmat, n, height, x, y)

                tadscores.update({chr:smat})
                chrdeltas.update({chr:gmat})
                chrbetas.update({chr:bmat})
        if verbose: 
                sys.stderr.write(time.strftime('\033[F\033[K[%H:%M:%S] Precomputing parameters completed\n', time.localtime()))
        # Background calculation
        for chr_index in numba.prange(len(chrs)):
                chr = chrs[chr_index]
                if verbose: 
                        if chr_index > 0:
                                sys.stderr.write('\033[F\033[K')
                        sys.stderr.write(time.strftime('[%H:%M:%S] Precomputing background scores for ', time.localtime()) + chr + '\n')
                mat = mats[chr]
                n = mat.shape[0]
                smat = np.zeros((n,n))
                __update_smat(mat, backgrnd, smat, height)
                bakscores.update({chr:smat})
        
        if verbose: 
                sys.stderr.write(time.strftime('\033[F\033[K[%H:%M:%S] Precomputing background completed\n', time.localtime()))

        if sys.version_info[0] >= 3 and sys.version_info[1] >= 2 and num_threads > 1:
                import concurrent.futures
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_threads)
                futures = []
        else:
                executor = None

        for chr_index in range(len(chrs)):
                chr = chrs[chr_index]
                map_file = contact_map_path[chr_index]
                map_name = contact_map_name[chr_index]
                map_size = N[chr_index]
                min_size = 2
                t_lim = M
                T_lim = N[chr_index]

                mat = mats[chr]
                gmat = chrdeltas[chr]
                bmat = chrbetas[chr]
                bakmat = bakscores[chr]
                smat = tadscores[chr] - bakscores[chr]
                # Dynamic programing
                if executor:
                        # print('\n\nconcurrent ' + chr + '\n\n')
                        fs = executor.submit(__process_chromosome, chr_index, min_size, t_lim, T_lim, mat, backgrnd, gmat, bmat, bakmat, smat, parameters)
                        futures.append(fs)
                        # print(fs)
                else:
                        if verbose: sys.stderr.write(time.strftime('[%H:%M:%S] Running dynamic program for ', time.localtime()) + chr + '\n')
                        __process_chromosome(chr_index, min_size, t_lim, T_lim, mat, backgrnd, gmat, bmat, bakmat, smat, parameters)
        if executor: 
                concurrent.futures.wait(futures)

if __name__ == '__main__':
        main()
