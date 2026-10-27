/* ===== Ina'ilau Shop - Main JavaScript ===== */
/* This script controls mobile navigation, dress filtering, search bar, client portal, inquiry bag, and contact forms */

document.addEventListener('DOMContentLoaded', () => {

  // Mobile menu toggle
  const menuToggle = document.getElementById('menuToggle');
  const navMenu    = document.getElementById('navMenu');

  if (menuToggle && navMenu) {
    menuToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
    });

    // Close menu when a link is clicked
    navMenu.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        navMenu.classList.remove('open');
      });
    });
  }

  // Collection dress filtering by category pills (All, Long, Mid, Warm, Cool)
  const filterButtons = document.querySelectorAll('.filter-btn');
  const dressCards    = document.querySelectorAll('.dress-card');

  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filterValue = btn.getAttribute('data-filter');

      dressCards.forEach(card => {
        const categories = card.getAttribute('data-category') || '';
        if (filterValue === 'all' || categories.includes(filterValue)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Click or tap on dress card image to toggle alternate photo view
  document.querySelectorAll('.dress-image-container').forEach(container => {
    container.addEventListener('click', () => {
      const card = container.closest('.dress-card');
      if (card) {
        card.classList.toggle('show-secondary');
      }
    });
  });

  // Modal open and close helper functions
  const openModal = (overlay) => {
    if (overlay) {
      overlay.classList.add('active');
      overlay.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }
  };

  const closeModal = (overlay) => {
    if (overlay) {
      overlay.classList.remove('active');
      overlay.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
    }
  };

  // Close modals when clicking the dark backdrop
  document.querySelectorAll('.site-modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        closeModal(overlay);
      }
    });
  });

  // Close modals when pressing the Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.site-modal-overlay.active').forEach(closeModal);
    }
  });

  // ===== Search Bar Functionality =====
  const searchBtn         = document.getElementById('searchBtn');
  const searchOverlay     = document.getElementById('searchOverlay');
  const searchCloseBtn    = document.getElementById('searchCloseBtn');
  const searchInput       = document.getElementById('searchInput');
  const searchClearBtn    = document.getElementById('searchClearBtn');
  const searchStatusBar   = document.getElementById('searchStatusBar');
  const searchResultsGrid = document.getElementById('searchResultsGrid');
  const searchTagChips    = document.querySelectorAll('.search-tag-chip');

  // Extract dress catalog data for fast search indexing
  const catalog = Array.from(dressCards).map((card, idx) => {
    const name = card.querySelector('.dress-name')?.textContent.trim() || '';
    const style = card.querySelector('.dress-style-tag')?.textContent.trim() || '';
    const primaryImg = card.querySelector('.dress-image-primary')?.getAttribute('src') || '';
    const categories = card.getAttribute('data-category') || '';
    const inquireBtn = card.querySelector('.inquire-btn');
    const dressData = inquireBtn?.getAttribute('data-dress') || name;
    return { id: idx, card, name, style, primaryImg, categories, dressData };
  });

  // Render search results inside the search modal
  const renderSearchResults = (query) => {
    const q = query.trim().toLowerCase();
    searchResultsGrid.innerHTML = '';

    const matches = catalog.filter(item => {
      if (!q) return true;
      return item.name.toLowerCase().includes(q) ||
             item.style.toLowerCase().includes(q) ||
             item.categories.toLowerCase().includes(q) ||
             item.dressData.toLowerCase().includes(q);
    });

    if (searchClearBtn) {
      searchClearBtn.style.display = q ? 'flex' : 'none';
    }

    if (matches.length === 0) {
      searchStatusBar.textContent = `No dresses found matching "${query}". Try searching "Aimata", "Mina", or "Mid".`;
      return;
    }

    searchStatusBar.textContent = q
      ? `Found ${matches.length} dress${matches.length === 1 ? '' : 'es'} matching "${query}"`
      : `Showing all 11 handcrafted dresses`;

    matches.forEach(item => {
      const itemEl = document.createElement('div');
      itemEl.className = 'search-result-item';
      itemEl.innerHTML = `
        <img src="${item.primaryImg}" alt="${item.name}" class="search-result-img" loading="lazy" />
        <div class="search-result-info">
          <span class="search-result-title">${item.name}</span>
          <span class="search-result-cut">${item.style}</span>
          <span class="search-result-action">View in Collection &rarr;</span>
        </div>
      `;

      // Clicking a search item scrolls to that dress in the main collection grid
      itemEl.addEventListener('click', () => {
        closeModal(searchOverlay);

        // Reset collection filter so all dresses are visible
        filterButtons.forEach(b => b.classList.toggle('active', b.getAttribute('data-filter') === 'all'));
        dressCards.forEach(c => c.style.display = 'flex');

        item.card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        item.card.style.boxShadow = '0 0 0 3px var(--color-terracotta)';
        setTimeout(() => {
          item.card.style.boxShadow = '';
        }, 2200);
      });

      searchResultsGrid.appendChild(itemEl);
    });
  };

  if (searchBtn && searchOverlay) {
    searchBtn.addEventListener('click', () => {
      openModal(searchOverlay);
      if (searchInput) {
        searchInput.value = '';
        renderSearchResults('');
        setTimeout(() => searchInput.focus(), 80);
      }
    });

    if (searchCloseBtn) {
      searchCloseBtn.addEventListener('click', () => closeModal(searchOverlay));
    }

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        renderSearchResults(e.target.value);
      });
    }

    if (searchClearBtn) {
      searchClearBtn.addEventListener('click', () => {
        searchInput.value = '';
        renderSearchResults('');
        searchInput.focus();
      });
    }

    // Quick tag chips in search modal
    searchTagChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        if (searchInput && query) {
          searchInput.value = query;
          renderSearchResults(query);
          searchInput.focus();
        }
      });
    });
  }

  // ===== Customer Account Portal Functionality =====
  const accountBtn      = document.getElementById('accountBtn');
  const accountOverlay  = document.getElementById('accountOverlay');
  const accountCloseBtn = document.getElementById('accountCloseBtn');
  const accountTabs     = document.querySelectorAll('.account-tab-btn');
  const accountPanels   = document.querySelectorAll('.account-panel');

  if (accountBtn && accountOverlay) {
    accountBtn.addEventListener('click', () => openModal(accountOverlay));

    if (accountCloseBtn) {
      accountCloseBtn.addEventListener('click', () => closeModal(accountOverlay));
    }

    // Tab switching inside client portal
    accountTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        accountTabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        accountPanels.forEach(p => p.classList.remove('active'));

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');

        const targetPanel = document.getElementById(tab.getAttribute('data-tab'));
        if (targetPanel) {
          targetPanel.classList.add('active');
        }
      });
    });

    // Inquiry Lookup Form Handler
    const inquiryLookupForm = document.getElementById('inquiryLookupForm');
    const lookupFeedback    = document.getElementById('lookupFeedback');
    if (inquiryLookupForm && lookupFeedback) {
      inquiryLookupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = document.getElementById('lookupEmail').value.trim();
        const ref = document.getElementById('lookupRef').value.trim() || 'INA-' + Math.floor(1000 + Math.random() * 9000);

        lookupFeedback.style.display = 'block';
        lookupFeedback.innerHTML = `
          <strong>Inquiry Found for ${email}</strong><br>
          Reference: <code>${ref}</code><br>
          Status: <em>Tailoring pattern consultation confirmed with our Polynesian artisan team.</em>
        `;
      });
    }

    // Client Sign-In Form Handler
    const clientSignInForm = document.getElementById('clientSignInForm');
    const signInFeedback   = document.getElementById('signInFeedback');
    if (clientSignInForm && signInFeedback) {
      clientSignInForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = document.getElementById('clientEmail').value.trim();

        signInFeedback.style.display = 'block';
        signInFeedback.innerHTML = `
          <strong>Welcome to Ina'ilau Client Services, ${email}!</strong><br>
          Your bespoke profile and saved Polynesian motifs are now synchronized.
        `;
      });
    }

    // Fitting Booking Form Handler
    const bookFittingForm = document.getElementById('bookFittingForm');
    const fittingFeedback = document.getElementById('fittingFeedback');
    if (bookFittingForm && fittingFeedback) {
      bookFittingForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const name  = document.getElementById('fittingClientName').value.trim();
        const dress = document.getElementById('fittingDressChoice').value;

        fittingFeedback.style.display = 'block';
        fittingFeedback.innerHTML = `
          <strong>Consultation Requested!</strong><br>
          Thank you, ${name}. Our head artisan will contact you to coordinate your private fitting for the <em>${dress}</em>.
        `;
        bookFittingForm.reset();
      });
    }
  }

  // ===== Inquiry Bag Drawer Functionality =====
  const bagBtn          = document.getElementById('bagBtn');
  const bagOverlay      = document.getElementById('bagOverlay');
  const bagCloseBtn     = document.getElementById('bagCloseBtn');
  const bagCount        = document.getElementById('bagCount');
  const bagDrawerBody   = document.getElementById('bagDrawerBody');

  const inquiryBag = [];

  // Update inquiry bag counter and render items
  const renderBag = () => {
    if (bagCount) {
      if (inquiryBag.length > 0) {
        bagCount.textContent = inquiryBag.length;
        bagCount.style.display = 'flex';
      } else {
        bagCount.style.display = 'none';
      }
    }

    if (!bagDrawerBody) return;

    if (inquiryBag.length === 0) {
      bagDrawerBody.innerHTML = `
        <div class="bag-empty-state">
          <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="color: var(--color-muted);">
            <path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"></path>
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <path d="M16 10a4 4 0 0 1-8 0"></path>
          </svg>
          <h3 class="bag-empty-title">Your Inquiry Bag is Empty</h3>
          <p class="bag-empty-text">Click "Inquire" on any dress in the collection to add it to your bespoke inquiry list.</p>
          <a href="#collection" class="btn btn-terracotta" id="browseFromBagBtn">BROWSE COLLECTION</a>
        </div>
      `;

      const browseBtn = document.getElementById('browseFromBagBtn');
      if (browseBtn) {
        browseBtn.addEventListener('click', () => closeModal(bagOverlay));
      }
      return;
    }

    let itemsHtml = '<div style="display: flex; flex-direction: column; gap: 0.85rem;">';
    inquiryBag.forEach((item, index) => {
      itemsHtml += `
        <div class="bag-item-card">
          <img src="${item.img}" alt="${item.name}" class="bag-item-img" />
          <div class="bag-item-info">
            <span class="bag-item-name">${item.name}</span>
            <span style="font-size: 0.8rem; color: var(--color-muted);">${item.style}</span>
            <div class="bag-item-action-row">
              <a href="#contact" class="bag-item-inquire-link" data-dress="${item.name}">Complete Inquiry &rarr;</a>
              <button type="button" class="bag-item-remove-btn" data-index="${index}">Remove</button>
            </div>
          </div>
        </div>
      `;
    });
    itemsHtml += '</div>';

    itemsHtml += `
      <div style="margin-top: 1rem; border-top: 1px solid var(--color-border); padding-top: 1.25rem;">
        <a href="#contact" class="btn btn-terracotta btn-full" id="submitAllInquiriesBtn">INQUIRE ABOUT ALL ITEMS</a>
      </div>
    `;

    bagDrawerBody.innerHTML = itemsHtml;

    // Attach click events for items in bag
    bagDrawerBody.querySelectorAll('.bag-item-remove-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.getAttribute('data-index'), 10);
        inquiryBag.splice(idx, 1);
        renderBag();
      });
    });

    bagDrawerBody.querySelectorAll('.bag-item-inquire-link').forEach(link => {
      link.addEventListener('click', () => {
        closeModal(bagOverlay);
        const dressSelect = document.getElementById('dressSelection');
        if (dressSelect) {
          dressSelect.value = link.getAttribute('data-dress');
        }
      });
    });

    const submitAllBtn = document.getElementById('submitAllInquiriesBtn');
    if (submitAllBtn) {
      submitAllBtn.addEventListener('click', () => {
        closeModal(bagOverlay);
        const dressSelect = document.getElementById('dressSelection');
        if (dressSelect) {
          dressSelect.value = inquiryBag.map(i => i.name).join(', ');
        }
      });
    }
  };

  if (bagBtn && bagOverlay) {
    bagBtn.addEventListener('click', () => {
      renderBag();
      openModal(bagOverlay);
    });

    if (bagCloseBtn) {
      bagCloseBtn.addEventListener('click', () => closeModal(bagOverlay));
    }
  }

  // Connect "Inquire" buttons to populate contact form and add dress to inquiry bag
  const inquireLinks = document.querySelectorAll('.inquire-btn, .inquire-link');
  const dressSelect  = document.getElementById('dressSelection');
  const messageInput = document.getElementById('messageText');

  inquireLinks.forEach(link => {
    link.addEventListener('click', () => {
      const dressName = link.getAttribute('data-dress');
      if (dressSelect && dressName) {
        dressSelect.value = dressName;
      }
      if (messageInput) {
        messageInput.focus();
      }

      // Add to inquiry bag if not already added
      const card = link.closest('.dress-card');
      if (card && dressName) {
        const primaryImg = card.querySelector('.dress-image-primary')?.getAttribute('src') || '';
        const style = card.querySelector('.dress-style-tag')?.textContent.trim() || '';
        if (!inquiryBag.some(i => i.name === dressName)) {
          inquiryBag.push({ name: dressName, style, img: primaryImg });
          renderBag();
        }
      }
    });
  });

  // Handle contact and inquiry form submission
  const inquiryForm  = document.getElementById('inquiryForm');
  const formFeedback = document.getElementById('formFeedback');

  if (inquiryForm && formFeedback) {
    inquiryForm.addEventListener('submit', (e) => {
      e.preventDefault();

      const nameVal  = document.getElementById('fullName').value.trim();
      const dressVal = document.getElementById('dressSelection').value;

      formFeedback.className = 'form-feedback success';
      formFeedback.style.display = 'block';

      if (dressVal) {
        formFeedback.textContent = `Thank you, ${nameVal}! Your inquiry for the ${dressVal} has been received. We will contact you within 24 hours with pricing and availability.`;
      } else {
        formFeedback.textContent = `Thank you, ${nameVal}! Your inquiry has been received. We will contact you within 24 hours.`;
      }

      inquiryForm.reset();

      setTimeout(() => {
        formFeedback.style.display = 'none';
      }, 7000);
    });
  }

});
