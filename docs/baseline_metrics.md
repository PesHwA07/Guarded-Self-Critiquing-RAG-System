# Baseline Metrics

This document tracks the baseline latency and sanity check results for the linear RAG pipeline (v1).

**Test Run Date:** 2026-07-06 12:54:54
**Total Queries:** 15
**Average Latency:** 4887.18 ms

## Query Results

### 1. How do I send a basic GET request?
- **Latency:** 678.57 ms
- **Sources Used:** [3]

**Answer:**

> You can send a basic GET request using `requests.get(url, auth=auth)` [Source 3].

---

### 2. How do I pass parameters in URLs?
- **Latency:** 287.05 ms
- **Sources Used:** [4]

**Answer:**

> You can pass parameters in URLs by using the `params` keyword argument and passing a dictionary of strings. For example: `requests.get('https://httpbin.org/get', params=payload)`. [Source 4]

---

### 3. How do I read the response content as JSON?
- **Latency:** 438.70 ms
- **Sources Used:** [3, 4]

**Answer:**

> Use the `json()` method or `Response.json` attribute to read the response content as JSON. Alternatively, use `Response.iter_content` with a chunk size to handle the content and then decode it as JSON.

---

### 4. How do I send a POST request with form data?
- **Latency:** 298.92 ms
- **Sources Used:** [1, 2]

**Answer:**

> Pass a dictionary to the `data` argument. If the dictionary has multiple values for each key, make `data` a dictionary with lists as values or a list of tuples. [Source 1, Source 2]

---

### 5. How do I set custom headers on my request?
- **Latency:** 504.03 ms
- **Sources Used:** [3, 1]

**Answer:**

> You can set custom headers on your request by passing a dict to the 'headers' parameter. For example: `r = requests.get(url, headers={'user-agent': 'my-app/0.0.1'})`. [Source 3] [Source 1]

---

### 6. How do I check the response status code?
- **Latency:** 478.13 ms
- **Sources Used:** [2, 3, 5]

**Answer:**

> You can check the response status code using r.status_code, where r is the Response object. Alternatively, you can use the built-in status code lookup object with requests.codes.ok, or use r.raise_for_status() to check for successful responses. [Source 2, Source 3, Source 5]

---

### 7. How do I handle cookies in a request?
- **Latency:** 672.78 ms
- **Sources Used:** [1, 3, 2, 5]

**Answer:**

> You can handle cookies in a request by using a RequestsCookieJar, passing a dictionary of cookies to the cookies parameter, or using a Session object with the prepare_request method. You can also manually add cookies to a Session object using the Cookie utility functions. [Source 1, Source 3, Source 2, Source 5]

---

### 8. How do I set a timeout on a request?
- **Latency:** 6595.51 ms
- **Sources Used:** [3, 4, 5]

**Answer:**

> You can set a timeout on a request by passing the timeout value to the get() method, either as a single value or as a tuple for separate connect and read timeouts. You can also set it to None to wait forever for a response. [Source 3, Source 4, Source 5]

---

### 9. What exception is raised if a request times out?
- **Latency:** 8510.44 ms
- **Sources Used:** [2]

**Answer:**

> A ~requests.exceptions.Timeout exception is raised if a request times out [Source 2].

---

### 10. How do I use sessions for connection pooling?
- **Latency:** 8644.36 ms
- **Sources Used:** [1]

**Answer:**

> You can use sessions for connection pooling by creating a Session object and making requests within it. This will automatically reuse the underlying TCP connection for requests to the same host, resulting in a performance increase.

---

### 11. How do I perform basic HTTP authentication?
- **Latency:** 7700.92 ms
- **Sources Used:** [4, 5]

**Answer:**

> You can perform basic HTTP authentication using the HTTPBasicAuth class from requests.auth, or by passing a tuple of ('user', 'pass') to the auth argument of a request method. [Source 4] [Source 5]

---

### 12. How do I upload a file in a POST request?
- **Latency:** 10635.49 ms
- **Sources Used:** [3, 4]

**Answer:**

> You can upload a file in a POST request by passing a dictionary of files to the 'files' parameter, where each key is the form field name and each value is a tuple of (filename, file content, content type). [Source 3] [Source 4]

---

### 13. How do I verify SSL certificates?
- **Latency:** 8751.59 ms
- **Sources Used:** [2, 4]

**Answer:**

> You can verify SSL certificates by setting the ``verify`` parameter to True, or by specifying a path to a CA_BUNDLE file or directory through the ``REQUESTS_CA_BUNDLE`` environment variable or the ``verify`` parameter. [Source 2], [Source 4].

---

### 14. How do I access the underlying urllib3 response?
- **Latency:** 9530.50 ms
- **Sources Used:** [1, 2]

**Answer:**

> You can access the underlying urllib3 response by using Response.raw <requests.Response.raw>.

---

### 15. How do I define a custom proxy?
- **Latency:** 9580.69 ms
- **Sources Used:** [1, 4]

**Answer:**

> To define a custom proxy, use the `scheme://hostname` form for the key, where proxy URLs must include the scheme. For example, `proxies = {'http://10.20.1.128': 'http://10.10.1.10:5323'}`. [Source 1] [Source 4]

---

