# Data Expansion Guide: 49 → 500-1000 Samples

**Context**: Current 49 samples failed to improve task-solving (0/20 benchmark). Need 500-1000 high-quality samples for v2.

**Sources**:
1. HumanEval (164 tasks) - use all
2. MBPP (1000+ tasks) - filter for simple ones
3. Synthetic generation (300-500 tasks) - fill gaps

---

## Category 1: Arithmetic & Math (Target: 50 samples)

**Current (9 samples)**: add, subtract, multiply, divide, modulo, square, cube, abs_value, power

**Add (41 samples)**:

### Basic Operations (10)
- `negate(n)` - Return -n
- `floor_divide(a, b)` - Return a // b
- `ceiling_divide(a, b)` - Return (a + b - 1) // b
- `sign(n)` - Return 1 if positive, -1 if negative, 0 if zero
- `clamp(n, min_val, max_val)` - Constrain n to [min_val, max_val]
- `lerp(a, b, t)` - Linear interpolation: a + (b - a) * t
- `avg(a, b)` - Return (a + b) / 2
- `gcd_simple(a, b)` - Already have, keep
- `lcm(a, b)` - Least common multiple
- `is_divisible(a, b)` - Return a % b == 0

### Number Theory (15)
- `is_prime(n)` - Already have
- `next_prime(n)` - Find next prime after n
- `prime_factors(n)` - Return list of prime factors
- `count_primes(n)` - Count primes up to n
- `is_perfect_square(n)` - Check if n = k^2 for some k
- `nth_prime(n)` - Return n-th prime number
- `digit_sum(n)` - Sum of digits in n
- `digit_count(n)` - Count digits in n
- `reverse_number(n)` - 123 → 321
- `is_palindrome_number(n)` - Check if n reads same forwards/backwards
- `collatz_steps(n)` - Count steps in Collatz sequence
- `sum_divisors(n)` - Sum of all divisors of n
- `is_perfect(n)` - n equals sum of its proper divisors
- `gcd_list(numbers)` - GCD of list of numbers
- `lcm_list(numbers)` - LCM of list of numbers

### Sequences (16)
- `factorial(n)` - Already have
- `fibonacci(n)` - Already have
- `fibonacci_list(n)` - Return first n Fibonacci numbers
- `lucas(n)` - Lucas sequence: L(n) = L(n-1) + L(n-2), L(0)=2, L(1)=1
- `triangular(n)` - n-th triangular number: n*(n+1)/2
- `square_sum(n)` - Sum of first n squares: 1^2 + 2^2 + ... + n^2
- `cube_sum(n)` - Sum of first n cubes
- `arithmetic_sequence(a, d, n)` - n-th term of arithmetic sequence
- `geometric_sequence(a, r, n)` - n-th term of geometric sequence
- `sum_arithmetic(a, d, n)` - Sum of first n terms
- `sum_geometric(a, r, n)` - Sum of first n terms
- `binomial(n, k)` - Binomial coefficient C(n,k)
- `catalan(n)` - n-th Catalan number
- `pascal_row(n)` - n-th row of Pascal's triangle
- `stirling(n, k)` - Stirling number of second kind
- `permutations(n, k)` - P(n,k) = n!/(n-k)!

**HumanEval tasks to use**: None directly (HumanEval focuses on algorithms, not pure math)

**MBPP tasks to filter**: Search for "factorial", "fibonacci", "prime", "gcd"

---

## Category 2: List Operations (Target: 100 samples)

**Current (13 samples)**: sum_list, reverse_list, find_max, remove_duplicates, length, first_element, last_element, contains, count_occurrences, append_item, concat_lists, filter_positive, filter_even, double_list

**Add (87 samples)**:

### Access & Extraction (10)
- `nth_element(items, n)` - Get n-th element (with bounds check)
- `slice_list(items, start, end)` - Return items[start:end]
- `take(items, n)` - First n elements
- `drop(items, n)` - All but first n elements
- `head(items)` - First element (alias for first_element)
- `tail(items)` - All but first element
- `init(items)` - All but last element
- `take_while(items, predicate)` - Take elements while condition true
- `drop_while(items, predicate)` - Drop elements while condition true
- `split_at(items, n)` - Split into (items[:n], items[n:])

### Aggregation (15)
- `sum_list(items)` - Already have
- `product(items)` - Product of all elements
- `average(items)` - Mean value
- `median(items)` - Middle value (sorted)
- `mode(items)` - Most frequent element
- `variance(items)` - Variance of values
- `std_dev(items)` - Standard deviation
- `min_list(items)` - Minimum element
- `max_list(items)` - Already have as find_max
- `range_list(items)` - max - min
- `count_if(items, predicate)` - Count elements satisfying condition
- `sum_if(items, predicate)` - Sum elements satisfying condition
- `all_satisfy(items, predicate)` - Check if all elements satisfy condition
- `any_satisfy(items, predicate)` - Check if any element satisfies condition
- `none_satisfy(items, predicate)` - Check if no elements satisfy condition

### Filtering & Selection (15)
- `filter_positive(items)` - Already have
- `filter_negative(items)` - Return only negative numbers
- `filter_even(items)` - Already have
- `filter_odd(items)` - Return only odd numbers
- `filter_greater_than(items, threshold)` - Elements > threshold
- `filter_less_than(items, threshold)` - Elements < threshold
- `filter_between(items, low, high)` - Elements in [low, high]
- `filter_not_none(items)` - Remove None values
- `partition(items, predicate)` - Split into (matching, non-matching)
- `take_even_indices(items)` - Elements at indices 0, 2, 4, ...
- `take_odd_indices(items)` - Elements at indices 1, 3, 5, ...
- `find_first(items, predicate)` - First element satisfying condition
- `find_last(items, predicate)` - Last element satisfying condition
- `find_all(items, predicate)` - All elements satisfying condition (alias for filter)
- `reject(items, predicate)` - Elements NOT satisfying condition

### Transformation (15)
- `double_list(items)` - Already have
- `triple_list(items)` - Multiply each by 3
- `square_list(items)` - Square each element
- `negate_list(items)` - Negate each element
- `abs_list(items)` - Absolute value of each
- `increment_list(items)` - Add 1 to each
- `map_function(items, func)` - Apply function to each (if we support lambdas)
- `cumsum(items)` - Cumulative sum: [1,2,3] → [1,3,6]
- `cumprod(items)` - Cumulative product
- `differences(items)` - Adjacent differences: [1,3,6] → [2,3]
- `pairwise(items)` - Adjacent pairs: [1,2,3] → [(1,2), (2,3)]
- `zip_lists(list1, list2)` - Zip two lists into pairs
- `unzip_pairs(pairs)` - Unzip pairs into two lists
- `intersperse(items, sep)` - Insert separator between elements
- `chunk(items, size)` - Split into chunks of size n

### Searching & Indexing (12)
- `linear_search(items, target)` - Already have
- `binary_search(items, target)` - Search in sorted list
- `find_index(items, target)` - Index of first occurrence
- `find_last_index(items, target)` - Index of last occurrence
- `find_all_indices(items, target)` - All indices of target
- `index_of_max(items)` - Index of maximum element
- `index_of_min(items)` - Index of minimum element
- `second_largest(items)` - Second largest element
- `kth_largest(items, k)` - k-th largest element
- `kth_smallest(items, k)` - k-th smallest element
- `closest_to(items, target)` - Element closest to target value
- `bisect_left(items, target)` - Leftmost insertion point in sorted list

### Reordering & Manipulation (20)
- `reverse_list(items)` - Already have
- `rotate_left(items, k)` - Rotate k positions left
- `rotate_right(items, k)` - Rotate k positions right
- `swap_elements(items, i, j)` - Swap items[i] and items[j]
- `shuffle_list(items, seed)` - Random permutation (deterministic with seed)
- `interleave(list1, list2)` - Alternate elements from two lists
- `flatten_once(nested)` - Flatten one level: [[1,2],[3]] → [1,2,3]
- `flatten_deep(nested)` - Flatten recursively
- `deduplicate(items)` - Remove duplicates (alias for remove_duplicates)
- `deduplicate_by_key(items, key_func)` - Remove duplicates using key function
- `group_by(items, key_func)` - Group elements by key
- `sort_list(items)` - Sort ascending
- `sort_descending(items)` - Sort descending
- `sort_by_key(items, key_func)` - Sort using custom key
- `stable_partition(items, predicate)` - Partition preserving relative order
- `unique_sorted(items)` - Sorted list with duplicates removed
- `sliding_window(items, size)` - Sliding windows of size n
- `transpose(matrix)` - Transpose 2D list
- `diagonal(matrix)` - Extract diagonal of 2D list
- `anti_diagonal(matrix)` - Extract anti-diagonal

**HumanEval tasks to use**:
- HumanEval/1: filter_by_prefix
- HumanEval/2: has_close_elements
- HumanEval/12: longest
- HumanEval/18: how_many_times
- HumanEval/23: strlen
- HumanEval/33: sort_third
- HumanEval/34: unique
- HumanEval/53: add_elements

**MBPP tasks to filter**: Search for "list", "array", "filter", "map", "sum"

---

## Category 3: String Operations (Target: 100 samples)

**Current (9 samples)**: count_vowels, is_palindrome, string_length, to_uppercase, to_lowercase, starts_with, ends_with, repeat_string, string_reverse

**Add (91 samples)**:

### Basic Operations (15)
- `char_at(text, n)` - Get character at position n
- `concat_strings(s1, s2)` - Concatenate two strings
- `substring(text, start, end)` - Extract substring
- `left(text, n)` - First n characters
- `right(text, n)` - Last n characters
- `trim(text)` - Remove leading/trailing whitespace
- `trim_left(text)` - Remove leading whitespace
- `trim_right(text)` - Remove trailing whitespace
- `pad_left(text, width, char)` - Pad on left to width
- `pad_right(text, width, char)` - Pad on right to width
- `center(text, width)` - Center string in width
- `ljust(text, width)` - Left-justify in width
- `rjust(text, width)` - Right-justify in width
- `zfill(text, width)` - Pad with zeros on left
- `capitalize(text)` - Capitalize first letter

### Checking & Validation (20)
- `is_empty(text)` - Check if empty or whitespace
- `is_alpha(text)` - All letters
- `is_digit(text)` - All digits
- `is_alnum(text)` - All letters or digits
- `is_lower(text)` - All lowercase
- `is_upper(text)` - All uppercase
- `is_title(text)` - Title case (each word capitalized)
- `is_palindrome(text)` - Already have
- `starts_with(text, prefix)` - Already have
- `ends_with(text, suffix)` - Already have
- `contains(text, substring)` - Check if substring present
- `contains_all(text, substrings)` - Check if all substrings present
- `contains_any(text, substrings)` - Check if any substring present
- `is_anagram(s1, s2)` - Check if anagrams
- `is_rotation(s1, s2)` - Check if s2 is rotation of s1
- `is_subsequence(sub, text)` - Check if sub is subsequence
- `is_whitespace(text)` - All whitespace
- `has_duplicates(text)` - Check for duplicate characters
- `is_balanced(text)` - Check balanced parentheses
- `is_valid_identifier(text)` - Valid Python identifier

### Counting & Statistics (15)
- `count_vowels(text)` - Already have
- `count_consonants(text)` - Count consonants
- `count_words(text)` - Word count
- `count_lines(text)` - Line count
- `count_char(text, char)` - Count occurrences of char
- `count_substring(text, sub)` - Count occurrences of substring
- `count_uppercase(text)` - Count uppercase letters
- `count_lowercase(text)` - Count lowercase letters
- `count_digits(text)` - Count digit characters
- `count_spaces(text)` - Count space characters
- `count_punctuation(text)` - Count punctuation marks
- `unique_chars(text)` - Count unique characters
- `char_frequency(text)` - Return dict of char frequencies
- `most_common_char(text)` - Most frequent character
- `least_common_char(text)` - Least frequent character

### Transformation (20)
- `to_uppercase(text)` - Already have
- `to_lowercase(text)` - Already have
- `swap_case(text)` - Swap upper/lower case
- `title_case(text)` - Convert to title case
- `camel_case(text)` - Convert to camelCase
- `snake_case(text)` - Convert to snake_case
- `kebab_case(text)` - Convert to kebab-case
- `reverse_string(text)` - Already have as string_reverse
- `reverse_words(text)` - Reverse word order
- `reverse_each_word(text)` - Reverse each word individually
- `remove_char(text, char)` - Remove all occurrences of char
- `remove_substring(text, sub)` - Remove all occurrences of substring
- `remove_vowels(text)` - Remove all vowels
- `remove_consonants(text)` - Remove all consonants
- `remove_digits(text)` - Remove all digits
- `remove_punctuation(text)` - Remove punctuation
- `remove_whitespace(text)` - Remove all whitespace
- `compress_whitespace(text)` - Collapse multiple spaces to one
- `remove_duplicates_str(text)` - Remove duplicate characters
- `interleave_strings(s1, s2)` - Alternate characters from two strings

### Splitting & Joining (10)
- `split_words(text)` - Split into words
- `split_lines(text)` - Split into lines
- `split_char(text, sep)` - Split on separator
- `split_n(text, sep, n)` - Split at most n times
- `join_words(words)` - Join with spaces
- `join_lines(lines)` - Join with newlines
- `join_with(items, sep)` - Join with custom separator
- `partition_string(text, sep)` - Split into (before, sep, after)
- `tokenize(text)` - Split into tokens
- `words_from_camel(text)` - "camelCase" → ["camel", "Case"]

### Searching & Replacing (11)
- `find_first_of(text, chars)` - Index of first occurrence of any char
- `find_last_of(text, chars)` - Index of last occurrence of any char
- `find_first_not_of(text, chars)` - Index of first char not in chars
- `index_of(text, substring)` - Index of substring
- `last_index_of(text, substring)` - Last index of substring
- `replace_first(text, old, new)` - Replace first occurrence
- `replace_last(text, old, new)` - Replace last occurrence
- `replace_all(text, old, new)` - Replace all occurrences
- `replace_chars(text, old_chars, new_char)` - Replace any of old_chars with new_char
- `translate(text, mapping)` - Translate using char mapping
- `expand_tabs(text, tabsize)` - Expand tabs to spaces

**HumanEval tasks to use**:
- HumanEval/5: intersperse
- HumanEval/10: make_palindrome
- HumanEval/20: find_closest_elements
- HumanEval/25: factorize
- HumanEval/27: flip_case
- HumanEval/28: concatenate
- HumanEval/43: pairs_sum_to_zero
- HumanEval/47: median
- HumanEval/58: common
- HumanEval/67: fruit_distribution

**MBPP tasks to filter**: Search for "string", "char", "word", "palindrome"

---

## Category 4: Algorithms (Target: 100 samples)

**Current (4 samples)**: bubble_sort, linear_search, flatten_list, gcd

**Add (96 samples)**:

### Sorting (15)
- `bubble_sort(arr)` - Already have
- `selection_sort(arr)` - Find min, swap, repeat
- `insertion_sort(arr)` - Insert each element into sorted portion
- `merge_sort(arr)` - Divide and conquer
- `quick_sort(arr)` - Partition around pivot
- `heap_sort(arr)` - Build heap, extract max
- `counting_sort(arr, k)` - Count occurrences (integers only)
- `radix_sort(arr)` - Sort by digits
- `bucket_sort(arr)` - Distribute into buckets
- `is_sorted(arr)` - Check if sorted
- `is_sorted_desc(arr)` - Check if sorted descending
- `sort_indices(arr)` - Return indices that would sort array
- `stable_sort(arr)` - Sort preserving equal element order
- `partial_sort(arr, k)` - Sort first k elements
- `nth_element(arr, n)` - Find n-th smallest (quick select)

### Searching (10)
- `linear_search(arr, target)` - Already have
- `binary_search(arr, target)` - Already have (if not, add)
- `binary_search_first(arr, target)` - First occurrence in sorted array
- `binary_search_last(arr, target)` - Last occurrence
- `exponential_search(arr, target)` - Binary search with exponential probe
- `jump_search(arr, target)` - Jump by √n steps
- `interpolation_search(arr, target)` - Estimate position
- `ternary_search(arr, target)` - Divide into 3 parts
- `find_peak(arr)` - Find local maximum
- `find_valley(arr)` - Find local minimum

### Two Pointers (10)
- `two_sum(arr, target)` - Find pair summing to target
- `three_sum(arr, target)` - Find triplet summing to target
- `remove_duplicates_sorted(arr)` - Remove duplicates from sorted array
- `merge_sorted(arr1, arr2)` - Merge two sorted arrays
- `intersection_sorted(arr1, arr2)` - Intersection of sorted arrays
- `union_sorted(arr1, arr2)` - Union of sorted arrays
- `is_subsequence_arr(sub, arr)` - Check if sub is subsequence
- `longest_common_prefix(strings)` - Longest common prefix
- `container_with_most_water(heights)` - Max water area
- `trapping_rain_water(heights)` - Total trapped water

### Sliding Window (10)
- `max_sum_subarray(arr, k)` - Max sum of k consecutive elements
- `min_sum_subarray(arr, k)` - Min sum of k consecutive elements
- `max_avg_subarray(arr, k)` - Max average of k consecutive
- `longest_substring_k_distinct(s, k)` - Longest with k distinct chars
- `min_window_substring(s, t)` - Smallest window containing all chars of t
- `longest_repeating_char(s, k)` - Longest with at most k replacements
- `max_consecutive_ones(arr, k)` - Max 1s after k flips
- `subarray_sum_equals(arr, target)` - Count subarrays summing to target
- `longest_subarray_sum(arr, target)` - Longest subarray with sum
- `shortest_subarray_sum(arr, target)` - Shortest subarray with sum >= target

### Recursion (15)
- `factorial_recursive(n)` - Recursive factorial
- `fibonacci_recursive(n)` - Recursive Fibonacci
- `power_recursive(base, exp)` - Already have as power
- `sum_recursive(arr)` - Sum array recursively
- `reverse_recursive(arr)` - Reverse array recursively
- `palindrome_recursive(s)` - Check palindrome recursively
- `tower_of_hanoi(n, from_rod, to_rod, aux_rod)` - Tower of Hanoi moves
- `all_subsets(arr)` - Generate all subsets (power set)
- `all_permutations(arr)` - Generate all permutations
- `all_combinations(arr, k)` - Generate all k-combinations
- `generate_parentheses(n)` - All valid n-pair parentheses
- `letter_combinations(digits)` - Phone keypad combinations
- `word_break(s, wordDict)` - Check if can break into words
- `decode_ways(s)` - Count ways to decode string
- `climb_stairs(n)` - Ways to climb n stairs (1 or 2 at a time)

### Dynamic Programming (15)
- `fibonacci_dp(n)` - Fibonacci with memoization
- `longest_increasing_subsequence(arr)` - Length of LIS
- `longest_common_subsequence(s1, s2)` - Length of LCS
- `edit_distance(s1, s2)` - Levenshtein distance
- `coin_change(coins, amount)` - Min coins to make amount
- `coin_change_ways(coins, amount)` - Number of ways
- `knapsack_01(weights, values, capacity)` - 0/1 knapsack
- `unbounded_knapsack(weights, values, capacity)` - Unbounded knapsack
- `partition_equal_subset(arr)` - Check if can partition into equal sums
- `max_subarray_sum(arr)` - Kadane's algorithm
- `max_product_subarray(arr)` - Max product of contiguous subarray
- `house_robber(arr)` - Max money without adjacent houses
- `unique_paths(m, n)` - Paths in m×n grid
- `min_path_sum(grid)` - Min sum path in grid
- `triangle_min_path(triangle)` - Min path sum in triangle

### Greedy (10)
- `activity_selection(starts, ends)` - Max non-overlapping activities
- `fractional_knapsack(weights, values, capacity)` - Fractional knapsack
- `job_sequencing(jobs, deadlines, profits)` - Max profit jobs
- `huffman_encoding(freqs)` - Huffman codes
- `interval_scheduling(intervals)` - Max non-overlapping intervals
- `gas_station(gas, cost)` - Can complete circular route
- `candy_distribution(ratings)` - Min candies with rating constraints
- `jump_game(arr)` - Can reach end of array
- `jump_game_min_jumps(arr)` - Min jumps to reach end
- `partition_labels(s)` - Partition into max substrings

### Math & Number Theory (11)
- `gcd(a, b)` - Already have
- `gcd_extended(a, b)` - Extended Euclidean (returns gcd, x, y)
- `lcm(a, b)` - Least common multiple
- `prime_sieve(n)` - Sieve of Eratosthenes
- `prime_factorization(n)` - Prime factors with powers
- `modular_exponentiation(base, exp, mod)` - (base^exp) % mod
- `modular_inverse(a, m)` - Inverse of a modulo m
- `chinese_remainder(remainders, moduli)` - Chinese remainder theorem
- `totient(n)` - Euler's totient function
- `fast_power(base, exp)` - Fast exponentiation
- `matrix_multiply(A, B)` - Matrix multiplication

**HumanEval tasks to use**:
- HumanEval/7: filter_by_substring
- HumanEval/11: string_xor
- HumanEval/15: string_sequence
- HumanEval/74: total_match
- HumanEval/80: is_happy
- HumanEval/85: add
- HumanEval/94: skjkasdkd
- HumanEval/97: multiply
- HumanEval/107: even_odd_palindrome
- HumanEval/113: odd_count

**MBPP tasks to filter**: Search for "sort", "search", "dynamic", "recursion"

---

## Category 5: Data Structures (Target: 50 samples)

**Current (0 samples)**

**Add (50 samples)**:

### Stacks (5)
- `stack_push(stack, item)` - Push onto stack
- `stack_pop(stack)` - Pop from stack
- `stack_peek(stack)` - Peek top element
- `stack_is_empty(stack)` - Check if empty
- `balanced_brackets(s)` - Check balanced using stack

### Queues (5)
- `queue_enqueue(queue, item)` - Add to queue
- `queue_dequeue(queue)` - Remove from queue
- `queue_front(queue)` - Peek front element
- `queue_is_empty(queue)` - Check if empty
- `circular_queue_next(queue, ptr, size)` - Next pointer in circular queue

### Linked Lists (10)
- `ll_insert_head(head, value)` - Insert at head
- `ll_insert_tail(head, value)` - Insert at tail
- `ll_delete_node(head, value)` - Delete first occurrence
- `ll_reverse(head)` - Reverse linked list
- `ll_find_middle(head)` - Find middle node
- `ll_detect_cycle(head)` - Detect cycle (Floyd's)
- `ll_remove_cycle(head)` - Remove cycle
- `ll_merge_sorted(l1, l2)` - Merge two sorted lists
- `ll_kth_from_end(head, k)` - Find k-th from end
- `ll_is_palindrome(head)` - Check if palindrome

### Trees (15)
- `tree_height(root)` - Height of binary tree
- `tree_size(root)` - Number of nodes
- `tree_sum(root)` - Sum of all nodes
- `tree_max(root)` - Maximum node value
- `tree_inorder(root)` - Inorder traversal
- `tree_preorder(root)` - Preorder traversal
- `tree_postorder(root)` - Postorder traversal
- `tree_level_order(root)` - Level order (BFS)
- `tree_is_bst(root)` - Check if BST
- `bst_insert(root, value)` - Insert into BST
- `bst_search(root, value)` - Search in BST
- `bst_delete(root, value)` - Delete from BST
- `bst_min(root)` - Minimum value in BST
- `bst_max(root)` - Maximum value in BST
- `bst_kth_smallest(root, k)` - k-th smallest in BST

### Graphs (10)
- `graph_dfs(graph, start)` - Depth-first search
- `graph_bfs(graph, start)` - Breadth-first search
- `graph_has_path(graph, start, end)` - Check if path exists
- `graph_shortest_path(graph, start, end)` - BFS shortest path
- `graph_is_bipartite(graph)` - Check if bipartite
- `graph_detect_cycle(graph)` - Detect cycle
- `graph_topological_sort(graph)` - Topological sort (DAG)
- `graph_connected_components(graph)` - Count components
- `dijkstra(graph, start)` - Shortest paths from start
- `bellman_ford(graph, start)` - Shortest paths (negative weights)

### Hash Tables & Sets (5)
- `two_sum_hash(arr, target)` - Two sum using hash
- `group_anagrams(words)` - Group anagrams using hash
- `first_unique_char(s)` - First non-repeating char
- `longest_substring_no_repeat(s)` - Longest with unique chars
- `subarray_sum_hash(arr, target)` - Subarray with given sum (hash)

**HumanEval tasks to use**:
- HumanEval/35: max_element
- HumanEval/42: incr_list
- HumanEval/54: same_chars
- HumanEval/78: hex_key

**MBPP tasks to filter**: Search for "stack", "queue", "tree", "graph"

---

## Category 6: Logic & Conditionals (Target: 50 samples)

**Current (6 samples)**: is_even, is_odd, is_positive, is_negative, max_value, min_value

**Add (44 samples)**:

### Comparison & Range (10)
- `in_range(n, low, high)` - Check if low <= n <= high
- `out_of_range(n, low, high)` - Check if n < low or n > high
- `compare_three(a, b, c)` - Return ordering of three numbers
- `is_between(x, a, b)` - Check if x between a and b
- `clamp_range(n, low, high)` - Clamp to [low, high]
- `is_close(a, b, tolerance)` - Check if |a-b| < tolerance
- `sign_of_product(a, b)` - Sign of a*b without multiplying
- `max_of_three(a, b, c)` - Max of three values
- `min_of_three(a, b, c)` - Min of three values
- `median_of_three(a, b, c)` - Median of three values

### Boolean Logic (10)
- `all_true(bools)` - Check if all true
- `any_true(bools)` - Check if any true
- `none_true(bools)` - Check if all false
- `exactly_one_true(bools)` - Check if exactly one true
- `xor(a, b)` - Exclusive or
- `implies(a, b)` - Logical implication a→b
- `nand(a, b)` - Not and
- `nor(a, b)` - Not or
- `majority(bools)` - Check if more than half true
- `parity(bools)` - Check if odd number true

### Conditional Flows (10)
- `absolute_difference(a, b)` - |a - b|
- `symmetric_difference(a, b, c)` - Elements in exactly one set
- `fizzbuzz(n)` - Return "Fizz", "Buzz", "FizzBuzz", or str(n)
- `leap_year(year)` - Check if leap year
- `day_of_week(day, month, year)` - Calculate day of week
- `grade_letter(score)` - Convert score to letter grade
- `tax_bracket(income)` - Determine tax bracket
- `shipping_cost(weight, distance)` - Calculate shipping cost
- `discount_price(price, quantity)` - Apply volume discount
- `bmi_category(weight, height)` - BMI category

### Pattern Matching (10)
- `match_pattern(text, pattern)` - Simple pattern matching
- `wildcard_match(text, pattern)` - Match with * and ?
- `regex_match_simple(text, pattern)` - Very simple regex
- `starts_and_ends(text, prefix, suffix)` - Check both
- `contains_all_chars(text, chars)` - Check if all chars present
- `contains_any_chars(text, chars)` - Check if any char present
- `validate_email_simple(email)` - Basic email validation
- `validate_phone_simple(phone)` - Basic phone validation
- `validate_password(password, rules)` - Password rules validation
- `validate_credit_card_luhn(card)` - Luhn algorithm

### Edge Case Handling (4)
- `safe_divide(a, b)` - Return a/b or 0 if b=0
- `safe_sqrt(n)` - Return sqrt(n) or 0 if negative
- `safe_index(arr, i)` - Return arr[i] or None if out of bounds
- `safe_get(dict, key, default)` - Get from dict with default

**HumanEval tasks to use**:
- HumanEval/3: below_zero
- HumanEval/37: sort_even
- HumanEval/60: sum_to_n
- HumanEval/61: correct_bracketing

**MBPP tasks to filter**: Search for "check", "validate", "conditional"

---

## Category 7: Loops & Iteration (Target: 50 samples)

**Current (4 samples)**: range_sum, range_product, all_positive, any_negative

**Add (46 samples)**:

### Simple Loops (10)
- `sum_n(n)` - Sum 1 to n
- `sum_evens(n)` - Sum even numbers 1 to n
- `sum_odds(n)` - Sum odd numbers 1 to n
- `product_n(n)` - Product 1 to n
- `count_down(n)` - Return list from n to 1
- `count_up(n)` - Return list from 1 to n
- `evens_up_to(n)` - Even numbers up to n
- `odds_up_to(n)` - Odd numbers up to n
- `multiples_of(k, n)` - Multiples of k up to n
- `powers_of_two(n)` - Powers of 2 up to 2^n

### Nested Loops (10)
- `multiplication_table(n)` - n×n multiplication table
- `all_pairs(arr1, arr2)` - All pairs from two arrays
- `matrix_sum(matrix)` - Sum of 2D matrix
- `matrix_transpose(matrix)` - Transpose matrix
- `matrix_diagonal(matrix)` - Diagonal elements
- `matrix_anti_diagonal(matrix)` - Anti-diagonal elements
- `flatten_2d(matrix)` - Flatten 2D to 1D
- `unflatten(arr, rows, cols)` - Reshape 1D to 2D
- `pascal_triangle(n)` - First n rows of Pascal's triangle
- `spiral_matrix(n)` - n×n spiral matrix

### Iteration Patterns (10)
- `foreach_print(items)` - Print each item
- `foreach_transform(items, func)` - Apply function to each
- `map_list(items, func)` - Map function over list
- `filter_list(items, predicate)` - Filter list by predicate
- `reduce_list(items, func, init)` - Reduce list
- `zip_multiple(lists)` - Zip multiple lists
- `enumerate_list(items)` - Return (index, item) pairs
- `grouped(items, n)` - Group into n-sized chunks
- `pairwise_apply(items, func)` - Apply to adjacent pairs
- `scan(items, func, init)` - Cumulative reduce

### Loop Control (8)
- `find_first_multiple(k, predicate)` - First multiple satisfying condition
- `collect_until(items, predicate)` - Collect until condition
- `skip_until(items, predicate)` - Skip until condition
- `take_until(items, predicate)` - Take until condition
- `process_with_break(items, limit)` - Process with early exit
- `continue_on_condition(items, predicate)` - Skip items conditionally
- `loop_with_counter(n, step)` - Count with custom step
- `infinite_sequence_first_n(generator, n)` - First n from generator

### Accumulation (8)
- `running_sum(items)` - Running sum
- `running_max(items)` - Running maximum
- `running_min(items)` - Running minimum
- `running_product(items)` - Running product
- `running_average(items)` - Running average
- `alternating_sum(items)` - +item[0] - item[1] + item[2] - ...
- `weighted_sum(values, weights)` - Sum of value*weight
- `dot_product(v1, v2)` - Vector dot product

**HumanEval tasks to use**:
- HumanEval/6: parse_nested_parens
- HumanEval/24: largest_divisor
- HumanEval/31: is_prime
- HumanEval/52: below_threshold

**MBPP tasks to filter**: Search for "loop", "iterate", "accumulate"

---

## Summary Table

| Category | Current | Target | To Add | Priority |
|----------|---------|--------|--------|----------|
| Math | 9 | 50 | 41 | Medium |
| List Ops | 13 | 100 | 87 | **High** |
| String Ops | 9 | 100 | 91 | **High** |
| Algorithms | 4 | 100 | 96 | **High** |
| Data Structures | 0 | 50 | 50 | Medium |
| Logic | 6 | 50 | 44 | Low |
| Loops | 4 | 50 | 46 | Low |
| **TOTAL** | **49** | **500** | **455** | - |

---

## Data Collection Workflow

### Phase 1: HumanEval (1-2 hours)
1. Download HumanEval dataset from GitHub
2. Extract all 164 tasks
3. Convert format: HumanEval format → our JSONL format
4. Verify: Run all test cases to ensure solutions correct
5. Filter: Remove tasks >50 lines or requiring complex classes

**Expected yield**: 120-140 tasks

### Phase 2: MBPP (2-3 hours)
1. Download MBPP dataset
2. Filter for "simple" difficulty
3. Filter for <=30 lines of code
4. Convert format to JSONL
5. Verify solutions

**Expected yield**: 200-300 tasks

### Phase 3: Synthetic Generation (3-5 hours)
1. Use GPT-4 to generate missing categories
2. Prompt template:
   ```
   Generate 10 Python function tasks with:
   - Function signature with type hints
   - Docstring describing task
   - Solution implementation
   - Test cases

   Category: [e.g., "list manipulation"]
   Difficulty: Simple (5-15 lines)
   ```
3. Validate all generated solutions
4. Manual review for quality

**Expected yield**: 100-200 tasks

### Phase 4: Consolidation (1 hour)
1. Merge all sources into single JSONL
2. Deduplicate similar tasks
3. Sort by category and difficulty
4. Split into train (450) / val (50)

---

## Quality Checks

Before using data for training:

1. **Syntactic correctness**: All solutions must parse
2. **Test coverage**: All solutions pass provided tests
3. **Diversity**: No near-duplicates (check via embedding similarity)
4. **Length distribution**: Mix of short (5-10 lines) and medium (10-30 lines)
5. **Category balance**: Each category has 50-100 samples

---

## Next Steps

1. **Decide scale**: 500 or 1000 samples?
   - 500: Fast iteration, lower quality ceiling
   - 1000: Slower but better coverage

2. **Prioritize categories**:
   - Start with: List Ops, String Ops, Algorithms (high value)
   - Later add: Data Structures, Math, Logic, Loops

3. **Tool selection**:
   - Manual curation (HumanEval/MBPP): High quality, slow
   - GPT-4 generation: Fast, requires validation

**Recommended**: Start with 500 samples (HumanEval + filtered MBPP), test v2 model, then expand to 1000 if needed.
